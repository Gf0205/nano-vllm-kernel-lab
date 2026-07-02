
// flash_attn_v2.cu
// 修复版：每个warp负责一行query，彻底消除冗余计算
//
// 设计：
//   grid: (num_heads, ceil(seq_len / WARPS_PER_BLOCK))
//   block: WARPS_PER_BLOCK * 32 个线程
//   每个warp处理一行query的完整attention计算
//
// 关键改变：
//   acc从 float[BLOCK_Q][HEAD_DIM] → float[D_PER_THREAD]（每线程只存HEAD_DIM/32个元素）
//   scores从 float[BLOCK_Q][BLOCK_KV] → 在循环里算完即用，不存寄存器

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <float.h>
#include <stdio.h>

// ---- warp内reduction（复用之前写的）----
__device__ __forceinline__ float warp_reduce_sum(float v) {
    v += __shfl_down_sync(0xFFFFFFFF, v, 16);
    v += __shfl_down_sync(0xFFFFFFFF, v, 8);
    v += __shfl_down_sync(0xFFFFFFFF, v, 4);
    v += __shfl_down_sync(0xFFFFFFFF, v, 2);
    v += __shfl_down_sync(0xFFFFFFFF, v, 1);
    return v;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
    v = fmaxf(v, __shfl_down_sync(0xFFFFFFFF, v, 16));
    v = fmaxf(v, __shfl_down_sync(0xFFFFFFFF, v, 8));
    v = fmaxf(v, __shfl_down_sync(0xFFFFFFFF, v, 4));
    v = fmaxf(v, __shfl_down_sync(0xFFFFFFFF, v, 2));
    v = fmaxf(v, __shfl_down_sync(0xFFFFFFFF, v, 1));
    return v;
}

// ============================================================
// FlashAttention V2 Kernel
//
// 模板参数：
//   HEAD_DIM:       head维度，必须是32的倍数（≤128）
//   WARPS_PER_BLOCK: 每个block处理几行query（同时也是block内warp数）
//
// 线程分工：
//   每个warp（32线程）负责一行query
//   lane_id（0-31）决定这个线程负责head_dim里的哪些维度
//   每个线程负责 HEAD_DIM/32 个维度
//   例：HEAD_DIM=64，每个线程负责2个维度
// ============================================================
template<int HEAD_DIM, int WARPS_PER_BLOCK>
__global__ void flash_attn_v2_kernel(
    const __half* __restrict__ Q,   // [seq, heads, dim]
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half* O,
    int seq_len,
    int num_heads,
    float scale
) {
    // ---- 线程身份 ----
    int head_id    = blockIdx.x;                    // 第几个head
    int warp_id    = threadIdx.x / 32;              // block内第几个warp
    int lane_id    = threadIdx.x % 32;              // warp内第几个线程（0-31）

    // 当前warp负责第几行query
    // blockIdx.y * WARPS_PER_BLOCK + warp_id
    int q_row = blockIdx.y * WARPS_PER_BLOCK + warp_id;
    if (q_row >= seq_len) return;

    // ---- 常量 ----
    // 每个线程负责head_dim里的几个维度
    // HEAD_DIM=64, 32线程 → 每线程2个维度
    // HEAD_DIM=128, 32线程 → 每线程4个维度
    constexpr int D_PER_THREAD = HEAD_DIM / 32;

    // ---- 加载Query（当前行）到寄存器 ----
    // Q[q_row, head_id, lane_id * D_PER_THREAD : (lane_id+1) * D_PER_THREAD]
    float q_reg[D_PER_THREAD];   // 寄存器数组，只有D_PER_THREAD=2-4个元素

    int q_base = q_row * num_heads * HEAD_DIM
               + head_id * HEAD_DIM
               + lane_id * D_PER_THREAD;

    #pragma unroll
    for (int d = 0; d < D_PER_THREAD; d++) {
        q_reg[d] = __half2float(Q[q_base + d]);
    }

    // ---- Online Softmax状态（每个warp的lane_id=0的线程维护）----
    // 但实际上每个线程都维护，用warp reduce同步
    float m_i = -FLT_MAX;   // 当前行最大score（标量）
    float l_i = 0.0f;       // 归一化系数（标量）

    // ---- 累加输出（寄存器里，不是shared memory）----
    float acc[D_PER_THREAD];
    #pragma unroll
    for (int d = 0; d < D_PER_THREAD; d++) acc[d] = 0.0f;

    // ---- KV循环 ----
    // Causal：只处理到 q_row 位置（含）
    int kv_end = q_row + 1;

    for (int kv_pos = 0; kv_pos < kv_end; kv_pos++) {
        // ---- 计算 Q[q_row] · K[kv_pos] ----
        // 每个线程计算自己负责的维度的部分积
        // 然后warp内reduce得到完整的点积

        int kv_base = kv_pos * num_heads * HEAD_DIM
                    + head_id * HEAD_DIM
                    + lane_id * D_PER_THREAD;

        float partial_dot = 0.0f;
        #pragma unroll
        for (int d = 0; d < D_PER_THREAD; d++) {
            float k_val = __half2float(K[kv_base + d]);
            partial_dot += q_reg[d] * k_val;
        }

        // Warp内reduce：32个线程的partial_dot求和 → 完整点积
        float score = warp_reduce_sum(partial_dot) * scale;
        // 此时所有线程都持有相同的score值（warp broadcast效果）
        // 注意：__shfl_down_sync后lane_id=0持有正确值，
        // 用__shfl_sync广播给所有线程
        score = __shfl_sync(0xFFFFFFFF, score, 0);

        // ---- Online Softmax更新 ----
        float m_new  = fmaxf(m_i, score);
        float alpha  = expf(m_i - m_new);
        float p      = expf(score - m_new);
        float l_new  = alpha * l_i + p;

        // ---- 更新累加器：acc = alpha * acc + p * V[kv_pos] ----
        #pragma unroll
        for (int d = 0; d < D_PER_THREAD; d++) {
            float v_val = __half2float(V[kv_base + d]);
            acc[d] = alpha * acc[d] + p * v_val;
        }

        m_i = m_new;
        l_i = l_new;
    }

    // ---- 最终归一化并写回 ----
    int o_base = q_row * num_heads * HEAD_DIM
               + head_id * HEAD_DIM
               + lane_id * D_PER_THREAD;

    #pragma unroll
    for (int d = 0; d < D_PER_THREAD; d++) {
        O[o_base + d] = __float2half(acc[d] / l_i);
    }
}


// ============================================================
// 封装函数
// ============================================================
void flash_attn_v2(
    const __half* Q, const __half* K, const __half* V, __half* O,
    int seq_len, int num_heads, int head_dim
) {
    float scale = 1.0f / sqrtf((float)head_dim);

    // 每个block处理 WARPS_PER_BLOCK 行query
    const int WARPS_PER_BLOCK = 4;
    const int THREADS = WARPS_PER_BLOCK * 32;  // 128线程

    // grid: (num_heads, ceil(seq_len / WARPS_PER_BLOCK))
    dim3 grid(num_heads, (seq_len + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    dim3 block(THREADS);

    if (head_dim == 64) {
        flash_attn_v2_kernel<64, WARPS_PER_BLOCK><<<grid, block>>>(
            Q, K, V, O, seq_len, num_heads, scale);
    } else if (head_dim == 128) {
        flash_attn_v2_kernel<128, WARPS_PER_BLOCK><<<grid, block>>>(
            Q, K, V, O, seq_len, num_heads, scale);
    }
    cudaDeviceSynchronize();
}


// ============================================================
// 正确性验证：和CPU标准Attention对比
// ============================================================
void cpu_attention(
    const __half* Q, const __half* K, const __half* V, __half* O,
    int seq_len, int num_heads, int head_dim
) {
    float scale = 1.0f / sqrtf((float)head_dim);

    for (int h = 0; h < num_heads; h++) {
        for (int i = 0; i < seq_len; i++) {
            // 计算softmax weights
            float scores[2048];  // 假设seq_len <= 2048
            float max_s = -FLT_MAX;

            for (int j = 0; j <= i; j++) {  // causal
                float s = 0;
                for (int d = 0; d < head_dim; d++) {
                    float q = __half2float(Q[i*num_heads*head_dim + h*head_dim + d]);
                    float k = __half2float(K[j*num_heads*head_dim + h*head_dim + d]);
                    s += q * k;
                }
                scores[j] = s * scale;
                max_s = fmaxf(max_s, scores[j]);
            }

            float sum_exp = 0;
            for (int j = 0; j <= i; j++) {
                scores[j] = expf(scores[j] - max_s);
                sum_exp += scores[j];
            }

            // 计算输出
            for (int d = 0; d < head_dim; d++) {
                float out = 0;
                for (int j = 0; j <= i; j++) {
                    float v = __half2float(V[j*num_heads*head_dim + h*head_dim + d]);
                    out += (scores[j] / sum_exp) * v;
                }
                O[i*num_heads*head_dim + h*head_dim + d] = __float2half(out);
            }
        }
    }
}


int main() {
    // ============================================================
    // Part 1：正确性验证
    // ============================================================
    const int SEQ = 64;    // 小一点，CPU参考实现能快点跑完
    const int H   = 4;
    const int D   = 64;
    int total = SEQ * H * D;

    __half *h_Q = new __half[total];
    __half *h_K = new __half[total];
    __half *h_V = new __half[total];
    __half *h_O_gpu = new __half[total];
    __half *h_O_cpu = new __half[total];

    srand(42);
    for (int i = 0; i < total; i++) {
        // 用小值初始化，避免fp16溢出
        h_Q[i] = __float2half(((float)rand()/RAND_MAX - 0.5f) * 0.1f);
        h_K[i] = __float2half(((float)rand()/RAND_MAX - 0.5f) * 0.1f);
        h_V[i] = __float2half(((float)rand()/RAND_MAX - 0.5f) * 0.1f);
    }

    __half *d_Q, *d_K, *d_V, *d_O;
    cudaMalloc(&d_Q, total * sizeof(__half));
    cudaMalloc(&d_K, total * sizeof(__half));
    cudaMalloc(&d_V, total * sizeof(__half));
    cudaMalloc(&d_O, total * sizeof(__half));

    cudaMemcpy(d_Q, h_Q, total*sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, total*sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, total*sizeof(__half), cudaMemcpyHostToDevice);

    // GPU计算
    flash_attn_v2(d_Q, d_K, d_V, d_O, SEQ, H, D);
    cudaMemcpy(h_O_gpu, d_O, total*sizeof(__half), cudaMemcpyDeviceToHost);

    // CPU参考计算
    printf("Running CPU reference (may take a few seconds)...\n");
    cpu_attention(h_Q, h_K, h_V, h_O_cpu, SEQ, H, D);

    // 比较
    float max_err = 0, avg_err = 0;
    for (int i = 0; i < total; i++) {
        float diff = fabsf(__half2float(h_O_gpu[i]) - __half2float(h_O_cpu[i]));
        max_err = fmaxf(max_err, diff);
        avg_err += diff;
    }
    avg_err /= total;

    printf("Correctness vs CPU reference:\n");
    printf("  max_err = %.4f\n", max_err);
    printf("  avg_err = %.6f\n", avg_err);
    printf("  Result: %s\n\n",
           max_err < 0.02f ? "✅ PASS" : "❌ FAIL");

    // ============================================================
    // Part 2：性能benchmark
    // ============================================================
    int seq_sizes[] = {128, 256, 512, 1024, 2048};
    const int BH = 8, BD = 64;

    printf("%-6s  %-12s  %-10s  %-10s\n",
           "seq", "Time(ms)", "TFLOPS", "vs_T4_peak");
    printf("─────────────────────────────────────────\n");

    for (int s : seq_sizes) {
        int tot = s * BH * BD;
        __half *dQ, *dK, *dV, *dO;
        cudaMalloc(&dQ, tot*sizeof(__half));
        cudaMalloc(&dK, tot*sizeof(__half));
        cudaMalloc(&dV, tot*sizeof(__half));
        cudaMalloc(&dO, tot*sizeof(__half));

        // 预热
        flash_attn_v2(dQ, dK, dV, dO, s, BH, BD);
        flash_attn_v2(dQ, dK, dV, dO, s, BH, BD);

        cudaEvent_t t0, t1;
        cudaEventCreate(&t0);
        cudaEventCreate(&t1);

        int runs = 50;
        cudaEventRecord(t0);
        for (int r = 0; r < runs; r++)
            flash_attn_v2(dQ, dK, dV, dO, s, BH, BD);
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);

        float ms;
        cudaEventElapsedTime(&ms, t0, t1);
        ms /= runs;

        // FLOPs：QK^T是 seq*seq*dim*2 per head，PV也是，共4倍
        // Causal：有效计算量减半，所以用2倍而不是4倍
        float flops  = 2.0f * 2.0f * (float)s * s * BD * BH;
        float tflops = flops / (ms * 1e-3f) / 1e12f;
        // T4 FP16峰值 ≈ 65 TFLOPS（Tensor Core），
        // 但我们用的是FP32计算，FP32峰值 ≈ 8.1 TFLOPS
        float pct_fp32_peak = tflops / 8.1f * 100.0f;

        printf("%-6d  %-12.3f  %-10.4f  %-10.1f%%\n",
               s, ms, tflops, pct_fp32_peak);

        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
        cudaFree(dQ); cudaFree(dK);
        cudaFree(dV); cudaFree(dO);
    }

    printf("\n注：使用FP32计算，对比FP32峰值(~8.1 TFLOPS)\n");
    printf("    如需更高性能：改用__half2做FP16计算 + Tensor Core\n");

    delete[] h_Q; delete[] h_K; delete[] h_V;
    delete[] h_O_gpu; delete[] h_O_cpu;
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    return 0;
}