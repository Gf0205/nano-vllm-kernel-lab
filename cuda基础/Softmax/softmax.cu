// softmax_t4_fixed.cu
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <float.h>

// ============================================================
// Warp / Block Reduce（纯片上洗牌网络）
// ============================================================
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

__device__ __forceinline__ float block_reduce_max(float val, float* smem) {
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;
    int num_warps = blockDim.x >> 5;

    val = warp_reduce_max(val);
    if (lane_id == 0) smem[warp_id] = val;
    __syncthreads();

    val = (threadIdx.x < num_warps) ? smem[threadIdx.x] : -FLT_MAX;
    if (warp_id == 0) val = warp_reduce_max(val);

    if (threadIdx.x == 0) smem[0] = val;
    __syncthreads();
    return smem[0];  
}

__device__ __forceinline__ float block_reduce_sum(float val, float* smem) {
    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;
    int num_warps = blockDim.x >> 5;

    val = warp_reduce_sum(val);
    if (lane_id == 0) smem[warp_id] = val;
    __syncthreads();

    val = (threadIdx.x < num_warps) ? smem[threadIdx.x] : 0.0f;
    if (warp_id == 0) val = warp_reduce_sum(val);

    if (threadIdx.x == 0) smem[0] = val;
    __syncthreads();
    return smem[0];
}

// ============================================================
// ★ 大 N 场景：ITEMS_PER_THREAD 编译期展开
// ============================================================
template<int BLOCK_SIZE, int ITEMS_PER_THREAD>
__global__ void softmax_optimized_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,   
    int M, int N
) {
    int row = blockIdx.x;
    if (row >= M) return;

    const int tid = threadIdx.x;
    const int num_v4 = N >> 2;  

    const float4* in_v4  = reinterpret_cast<const float4*>(input  + row * N);
    float4* out_v4 = reinterpret_cast<float4*>(output + row * N);

    float4 buf[ITEMS_PER_THREAD];

    // ═══════════ Pass 1: Load + Find Max ═══════════
    float local_max = -FLT_MAX;

    #pragma unroll    
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int idx = i * BLOCK_SIZE + tid;
        if (idx < num_v4) {
            buf[i] = in_v4[idx];
            // 👑 铁血修复点：全面回归正统的 fmaxf 硬件级内建指令
            local_max = fmaxf(local_max, buf[i].x);
            local_max = fmaxf(local_max, buf[i].y);
            local_max = fmaxf(local_max, buf[i].z);
            local_max = fmaxf(local_max, buf[i].w);
        }
    }

    __shared__ float smem[BLOCK_SIZE / 32];
    float row_max = block_reduce_max(local_max, smem);

    // ═══════════ Pass 2: Exp + Sum ═══════════
    float local_sum = 0.0f;

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int idx = i * BLOCK_SIZE + tid;
        if (idx < num_v4) {
            buf[i].x = __expf(buf[i].x - row_max);
            buf[i].y = __expf(buf[i].y - row_max);
            buf[i].z = __expf(buf[i].z - row_max);
            buf[i].w = __expf(buf[i].w - row_max);

            local_sum += buf[i].x + buf[i].y + buf[i].z + buf[i].w;
        }
    }

    float row_sum = block_reduce_sum(local_sum, smem);

    // 1 次 __fdividef 倒数求取大计
    float inv_sum = __fdividef(1.0f, row_sum);

    // ═══════════ Pass 3: Normalize + Store ═══════════
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int idx = i * BLOCK_SIZE + tid;
        if (idx < num_v4) {
            buf[i].x *= inv_sum;
            buf[i].y *= inv_sum;
            buf[i].z *= inv_sum;
            buf[i].w *= inv_sum;

            out_v4[idx] = buf[i];
        }
    }
}

// ============================================================
// ★ 小 N 场景专用（多行共享 Block，Warp级别大合并）
// ============================================================
template<int BLOCK_SIZE>
__global__ void softmax_multi_row_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int M, int N
) {
    int warp_id_in_block = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int num_warps = BLOCK_SIZE >> 5;

    int row = blockIdx.x * num_warps + warp_id_in_block;
    if (row >= M) return;

    const float4* in_v4  = reinterpret_cast<const float4*>(input  + row * N);
    float4* out_v4 = reinterpret_cast<float4*>(output + row * N);

    int num_v4 = N >> 2;

    float4 val = {0, 0, 0, 0};
    float local_max = -FLT_MAX;
    if (lane < num_v4) {
        val = in_v4[lane];
        // 👑 铁血修复点：多行小算子同步修正为正统 fmaxf
        local_max = fmaxf(fmaxf(val.x, val.y), fmaxf(val.z, val.w));
    }

    float row_max = warp_reduce_max(local_max);  

    float local_sum = 0.0f;
    if (lane < num_v4) {
        val.x = __expf(val.x - row_max);
        val.y = __expf(val.y - row_max);
        val.z = __expf(val.z - row_max);
        val.w = __expf(val.w - row_max);
        local_sum = val.x + val.y + val.z + val.w;
    }

    float row_sum = warp_reduce_sum(local_sum);
    float inv_sum = __fdividef(1.0f, row_sum);

    if (lane < num_v4) {
        val.x *= inv_sum;
        val.y *= inv_sum;
        val.z *= inv_sum;
        val.w *= inv_sum;
        out_v4[lane] = val;
    }
}

// ============================================================
// 统一调度器
// ============================================================
void softmax_dispatch(const float* input, float* output, int M, int N) {
    if (N <= 128) {
        constexpr int BS = 256;
        int rows_per_block = BS / 32;
        int grid = (M + rows_per_block - 1) / rows_per_block;
        softmax_multi_row_kernel<BS><<<grid, BS>>>(input, output, M, N);
    } else {
        constexpr int BS = 256;
        int items = (N / 4 + BS - 1) / BS;

        switch (items) {
            case 1:  softmax_optimized_kernel<BS, 1> <<<M, BS>>>(input, output, M, N); break;
            case 2:  softmax_optimized_kernel<BS, 2> <<<M, BS>>>(input, output, M, N); break;
            case 4:  softmax_optimized_kernel<BS, 4> <<<M, BS>>>(input, output, M, N); break;
            case 8:  softmax_optimized_kernel<BS, 8> <<<M, BS>>>(input, output, M, N); break;
            case 16: softmax_optimized_kernel<BS, 16><<<M, BS>>>(input, output, M, N); break;
            default: softmax_optimized_kernel<BS, 16><<<M, BS>>>(input, output, M, N); break;
        }
    }
}

int main() {
    const int M = 512, N = 1024;

    float *h_in      = new float[M * N];
    float *h_out_gpu = new float[M * N];
    float *h_out_cpu = new float[M * N];

    srand(42);
    for (int i = 0; i < M * N; i++) h_in[i] = (float)rand() / RAND_MAX - 0.5f;

    float *d_in, *d_out;
    cudaMalloc(&d_in,  M * N * sizeof(float));
    cudaMalloc(&d_out, M * N * sizeof(float));
    cudaMemcpy(d_in, h_in, M * N * sizeof(float), cudaMemcpyHostToDevice);

    softmax_dispatch(d_in, d_out, M, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out_gpu, d_out, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // CPU 参考
    for (int i = 0; i < M; i++) {
        float mx = -FLT_MAX;
        for (int j = 0; j < N; j++) mx = fmaxf(mx, h_in[i * N + j]);
        float s = 0;
        for (int j = 0; j < N; j++) {
            h_out_cpu[i * N + j] = expf(h_in[i * N + j] - mx);
            s += h_out_cpu[i * N + j];
        }
        for (int j = 0; j < N; j++) h_out_cpu[i * N + j] /= s;
    }

    float max_err = 0;
    for (int i = 0; i < M * N; i++)
        max_err = fmaxf(max_err, fabsf(h_out_gpu[i] - h_out_cpu[i]));
    printf("🚀 T4 Fused Multi-Strategy Softmax max_err=%.2e %s\n", max_err, max_err < 1e-4f ? "✅" : "❌");

    // 跑分大盘
    int row_sizes[] = {128, 1024, 2048, 4096};
    int test_M = 4096;

    printf("\n%-6s  %-10s  %-10s\n", "N", "Time(ms)", "T4-GB/s");
    printf("─────────────────────────────────\n");

    for (int n : row_sizes) {
        float *dI, *dO;
        cudaMalloc(&dI, test_M * n * sizeof(float));
        cudaMalloc(&dO, test_M * n * sizeof(float));

        cudaEvent_t t0, t1;
        cudaEventCreate(&t0);
        cudaEventCreate(&t1);

        softmax_dispatch(dI, dO, test_M, n);
        cudaDeviceSynchronize();

        int runs = 200;
        cudaEventRecord(t0);
        for (int r = 0; r < runs; r++)
            softmax_dispatch(dI, dO, test_M, n);
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);

        float ms;
        cudaEventElapsedTime(&ms, t0, t1);
        ms /= runs;

        // Fused寄存器架构只读1次写1次，有效显存吞吐 = 2 * M * N * 4 bytes
        float gbps = 2.0f * test_M * n * sizeof(float) / ms * 1e-6f;
        printf("%-6d  %-10.3f  %-10.1f (T4真实利用率: %.1f%%)\n",
               n, ms, gbps, gbps / 300.0f * 100);

        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
        cudaFree(dI);
        cudaFree(dO);
    }

    delete[] h_in; delete[] h_out_gpu; delete[] h_out_cpu;
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}