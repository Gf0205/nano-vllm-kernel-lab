#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

// ============================================================
// V1: 你的原版（保留对比）
// ============================================================
template<int BLOCK_SIZE>
__global__ void reduce_sum_v1(
    const float* __restrict__ input,
    float* output,
    int N
) {
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * BLOCK_SIZE + tid;
    const int stride = gridDim.x * BLOCK_SIZE;

    float thread_sum = 0.0f;
    const float4* input_v4 = reinterpret_cast<const float4*>(input);
    const int num_v4 = N >> 2;

    #pragma unroll(8)
    for (int i = idx; i < num_v4; i += stride) {
        float4 val = input_v4[i];
        thread_sum += val.x + val.y + val.z + val.w;
    }

    __shared__ float smem[BLOCK_SIZE];
    smem[tid] = thread_sum;
    __syncthreads();

    #pragma unroll
    for (int mask = BLOCK_SIZE >> 1; mask >= 32; mask >>= 1) {
        if (tid < mask) smem[tid] += smem[tid + mask];
        __syncthreads();
    }

    if (tid < 32) {
        float warp_sum = smem[tid];
        warp_sum += __shfl_xor_sync(0xffffffff, warp_sum, 16);
        warp_sum += __shfl_xor_sync(0xffffffff, warp_sum, 8);
        warp_sum += __shfl_xor_sync(0xffffffff, warp_sum, 4);
        warp_sum += __shfl_xor_sync(0xffffffff, warp_sum, 2);
        warp_sum += __shfl_xor_sync(0xffffffff, warp_sum, 1);
        if (tid == 0) output[blockIdx.x] = warp_sum;
    }
}

// ============================================================
// V2: 优化版
//
// 优化点：
//   1. 4 个独立累加器 → 打破 FADD 依赖链，ILP 4x
//   2. 每次循环处理 2 个 float4 = 32 Bytes → 减少循环开销
//   3. grid 大小对齐 SM 数量 → 均匀分配
//   4. Warp-first 规约 → 减少 smem 写入量
//   5. 处理尾部元素 → N % 4 也正确
//   6. atomicAdd 单 pass → 无需 CPU 二次规约
// ============================================================
template<int BLOCK_SIZE>
__global__ void reduce_sum_v2(
    const float* __restrict__ input,
    float*       __restrict__ output,
    int N
) {
    const int tid = threadIdx.x;

    // ── 优化1: 4 个独立累加器，打破依赖链 ──
    // 原版: thread_sum += val.x + val.y + val.z + val.w
    //   → 5 级串行 FADD（x+y → +z → +w → +sum）
    //
    // 优化: 4 路独立累加
    //   sum0 += val.x   (独立)
    //   sum1 += val.y   (独立)
    //   sum2 += val.z   (独立)
    //   sum3 += val.w   (独立)
    //   → 每路只有 1 级依赖，硬件可 4 路并行发射
    float sum0 = 0.f, sum1 = 0.f, sum2 = 0.f, sum3 = 0.f;

    const float4* input_v4 = reinterpret_cast<const float4*>(input);
    const int num_v4 = N >> 2;

    // grid-stride 索引
    int base = blockIdx.x * BLOCK_SIZE + tid;
    int stride = gridDim.x * BLOCK_SIZE;

    // ── 优化2: 每次处理 2 个 float4 = 32 Bytes ──
    // 减少循环控制指令占比，进一步提升 ILP
    int i = base;
    for (; i + stride < num_v4; i += stride * 2) {
        float4 v0 = input_v4[i];
        float4 v1 = input_v4[i + stride];
        // 8 个独立 FADD，硬件可全部并行
        sum0 += v0.x;  sum1 += v0.y;  sum2 += v0.z;  sum3 += v0.w;
        sum0 += v1.x;  sum1 += v1.y;  sum2 += v1.z;  sum3 += v1.w;
    }
    // 处理剩余的单个 float4
    if (i < num_v4) {
        float4 v0 = input_v4[i];
        sum0 += v0.x;  sum1 += v0.y;  sum2 += v0.z;  sum3 += v0.w;
    }

    // ── 优化5: 处理尾部（N 不是 4 的倍数时） ──
    if (tid == 0 && blockIdx.x == 0) {
        for (int j = num_v4 * 4; j < N; j++)
            sum0 += input[j];
    }

    // 合并 4 路
    float thread_sum = (sum0 + sum1) + (sum2 + sum3);

    // ── 优化4: Warp-first 规约 ──
    // 先在 warp 内用 shuffle 规约（32→1），
    // 再写 smem 做 block 级规约
    // 好处：smem 只需 BLOCK_SIZE/32 个槽位，写入量减少 8x
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);

    constexpr int NUM_WARPS = BLOCK_SIZE / 32;
    __shared__ float warp_sums[NUM_WARPS];

    int warp_id = tid >> 5;
    int lane_id = tid & 31;

    if (lane_id == 0)
        warp_sums[warp_id] = thread_sum;
    __syncthreads();

    // 最后一个 warp 做 warp 间规约
    if (tid < NUM_WARPS) {
        float val = warp_sums[tid];
        // NUM_WARPS = 8 (for BLOCK_SIZE=256)，只需 3 级
        #pragma unroll
        for (int offset = NUM_WARPS >> 1; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);

        // ── 优化6: atomicAdd 单 pass 输出 ──
        if (tid == 0)
            atomicAdd(output, val);
    }
}

int main() {
    const int N = 1 << 24;  // 16M
    const int BLOCK_SIZE = 256;

    // ── 优化3: grid 对齐 SM 数 ──
    // T4 = 40 SM，每 SM 最多 2048 线程 = 8 blocks of 256
    // 最优: 40 × 8 = 320 blocks (刚好占满所有 SM)
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    int sm_count = prop.multiProcessorCount;

    // 每 SM 允许的最大 block 数
    int max_blocks_per_sm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm, reduce_sum_v2<BLOCK_SIZE>, BLOCK_SIZE, 0);
    int num_blocks_v2 = sm_count * max_blocks_per_sm;

    printf("GPU: %s, %d SMs, grid=%d blocks (%.0f blocks/SM)\n",
           prop.name, sm_count, num_blocks_v2,
           (float)num_blocks_v2 / sm_count);

    // ── 分配 ──
    float *h_in = new float[N];
    for (int i = 0; i < N; i++) h_in[i] = 1.0f;

    float *d_in, *d_out_v1, *d_out_v2;
    const int num_blocks_v1 = 256;
    cudaMalloc(&d_in,     N * sizeof(float));
    cudaMalloc(&d_out_v1, num_blocks_v1 * sizeof(float));
    cudaMalloc(&d_out_v2, sizeof(float));     // V2 只需 1 个 float
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    // ── V1 正确性 ──
    reduce_sum_v1<BLOCK_SIZE><<<num_blocks_v1, BLOCK_SIZE>>>(d_in, d_out_v1, N);
    cudaDeviceSynchronize();
    float h_partial[256];
    cudaMemcpy(h_partial, d_out_v1, num_blocks_v1*sizeof(float), cudaMemcpyDeviceToHost);
    float v1_sum = 0;
    for (int i = 0; i < num_blocks_v1; i++) v1_sum += h_partial[i];
    printf("V1 sum: %.0f %s\n", v1_sum, fabsf(v1_sum-N)<1.f ? "✅" : "❌");

    // ── V2 正确性 ──
    cudaMemset(d_out_v2, 0, sizeof(float));  // ★ 必须清零（atomicAdd 累加）
    reduce_sum_v2<BLOCK_SIZE><<<num_blocks_v2, BLOCK_SIZE>>>(d_in, d_out_v2, N);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("❌ V2 error: %s\n", cudaGetErrorString(err));
    float v2_sum;
    cudaMemcpy(&v2_sum, d_out_v2, sizeof(float), cudaMemcpyDeviceToHost);
    printf("V2 sum: %.0f %s\n", v2_sum, fabsf(v2_sum-N)<1.f ? "✅" : "❌");

    // ── Benchmark ──
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    int runs = 200;
    float ms;

    // V1
    reduce_sum_v1<BLOCK_SIZE><<<num_blocks_v1, BLOCK_SIZE>>>(d_in, d_out_v1, N);
    cudaDeviceSynchronize();
    cudaEventRecord(t0);
    for (int r = 0; r < runs; r++)
        reduce_sum_v1<BLOCK_SIZE><<<num_blocks_v1, BLOCK_SIZE>>>(d_in, d_out_v1, N);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms, t0, t1);
    float v1_ms = ms / runs;
    float v1_gbps = (float)N * sizeof(float) / v1_ms * 1e-6f;

    // V2（注意：每次 run 前要 memset 清零）
    // 但 benchmark 不需要正确结果，只要测速，所以不清零也行
    reduce_sum_v2<BLOCK_SIZE><<<num_blocks_v2, BLOCK_SIZE>>>(d_in, d_out_v2, N);
    cudaDeviceSynchronize();
    cudaEventRecord(t0);
    for (int r = 0; r < runs; r++)
        reduce_sum_v2<BLOCK_SIZE><<<num_blocks_v2, BLOCK_SIZE>>>(d_in, d_out_v2, N);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms, t0, t1);
    float v2_ms = ms / runs;
    float v2_gbps = (float)N * sizeof(float) / v2_ms * 1e-6f;

    float peak_bw = 320.0f;  // T4
    printf("\n%-6s  %-10s  %-10s  %-12s\n", "Ver", "Time(ms)", "GB/s", "利用率");
    printf("──────────────────────────────────────────\n");
    printf("V1      %-10.3f  %-10.1f  %.1f%%\n", v1_ms, v1_gbps, v1_gbps/peak_bw*100);
    printf("V2      %-10.3f  %-10.1f  %.1f%% (%.1fx)\n",
           v2_ms, v2_gbps, v2_gbps/peak_bw*100, v1_ms/v2_ms);

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    delete[] h_in;
    cudaFree(d_in); cudaFree(d_out_v1); cudaFree(d_out_v2);
    return 0;
}