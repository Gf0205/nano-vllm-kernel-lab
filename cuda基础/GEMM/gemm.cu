#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

// ============================================================
// V2: Tiled GEMM（保留用于对比）
// ============================================================
template<int BM, int BN, int BK>
__global__ void gemm_v2_tiled(
    const float* __restrict__ A, const float* __restrict__ B,
    float* __restrict__ C, int M, int N, int K
) {
    __shared__ float smem_A[BM][BK];
    __shared__ float smem_B[BK][BN];
    int row = blockIdx.y * BM + threadIdx.y;
    int col = blockIdx.x * BN + threadIdx.x;
    float sum = 0.f;
    for (int tile = 0; tile < (K + BK - 1) / BK; tile++) {
        int a_col = tile * BK + threadIdx.x;
        int b_row = tile * BK + threadIdx.y;
        smem_A[threadIdx.y][threadIdx.x] = (row<M && a_col<K) ? A[row*K+a_col] : 0.f;
        smem_B[threadIdx.y][threadIdx.x] = (b_row<K && col<N) ? B[b_row*N+col] : 0.f;
        __syncthreads();
        for (int k = 0; k < BK; k++) sum += smem_A[threadIdx.y][k] * smem_B[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row*N+col] = sum;
}

// ============================================================
// V3: Register Blocking（上一版，保留对比）
// ============================================================
template<int BM, int BN, int BK, int TM, int TN>
__global__ void gemm_v3_reg_block(
    const float* __restrict__ A, const float* __restrict__ B,
    float* __restrict__ C, int M, int N, int K
) {
    constexpr int THREAD_M = BM / TM;
    constexpr int THREAD_N = BN / TN;
    constexpr int THREADS   = THREAD_M * THREAD_N;
    int tid = threadIdx.x;
    int ty  = tid / THREAD_N;
    int tx  = tid % THREAD_N;
    int row_start = blockIdx.y * BM;
    int col_start = blockIdx.x * BN;
    __shared__ float smem_A[BK][BM];
    __shared__ float smem_B[BK][BN];
    float reg_C[TM][TN] = {};
    float reg_A[TM], reg_B[TN];
    constexpr int A_EPT = BK * BM / THREADS;
    constexpr int B_EPT = BK * BN / THREADS;
    for (int tile_k = 0; tile_k < (K + BK - 1) / BK; tile_k++) {
        int k_base = tile_k * BK;
        #pragma unroll
        for (int i = 0; i < A_EPT; i++) {
            int flat = i * THREADS + tid;
            int k = flat / BM, m = flat % BM;
            int g_row = row_start + m, g_col = k_base + k;
            smem_A[k][m] = (g_row < M && g_col < K) ? A[g_row * K + g_col] : 0.f;
        }
        #pragma unroll
        for (int i = 0; i < B_EPT; i++) {
            int flat = i * THREADS + tid;
            int k = flat / BN, n = flat % BN;
            int g_row = k_base + k, g_col = col_start + n;
            smem_B[k][n] = (g_row < K && g_col < N) ? B[g_row * N + g_col] : 0.f;
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            #pragma unroll
            for (int m = 0; m < TM; m++) reg_A[m] = smem_A[k][ty * TM + m];
            #pragma unroll
            for (int n = 0; n < TN; n++) reg_B[n] = smem_B[k][tx * TN + n];
            #pragma unroll
            for (int m = 0; m < TM; m++)
                #pragma unroll
                for (int n = 0; n < TN; n++)
                    reg_C[m][n] += reg_A[m] * reg_B[n];
        }
        __syncthreads();
    }
    #pragma unroll
    for (int m = 0; m < TM; m++) {
        int g_row = row_start + ty * TM + m;
        if (g_row >= M) continue;
        #pragma unroll
        for (int n = 0; n < TN; n++) {
            int g_col = col_start + tx * TN + n;
            if (g_col < N) C[g_row * N + g_col] = reg_C[m][n];
        }
    }
}

// ── 预处理宏平替 Lambda ──
#define LOAD_TILE_MACRO(tile_k, buf_idx) \
{ \
    int k_base = (tile_k) * BK; \
    _Pragma("unroll") \
    for (int i = 0; i < A_F4_PER_THREAD; i++) { \
        int f4_idx = i * THREADS + tid; \
        int k      = f4_idx / A_VEC_PER_ROW; \
        int m_base = (f4_idx % A_VEC_PER_ROW) * VEC; \
        int g_col  = k_base + k; \
        _Pragma("unroll") \
        for (int v = 0; v < VEC; v++) { \
            int g_row = row_start + m_base + v; \
            smem_A[(buf_idx)][k][m_base + v] = \
                (g_row < M && g_col < K) ? A[g_row * K + g_col] : 0.f; \
        } \
    } \
    _Pragma("unroll") \
    for (int i = 0; i < B_F4_PER_THREAD; i++) { \
        int f4_idx = i * THREADS + tid; \
        int k      = f4_idx / (BN / VEC); \
        int n_base = (f4_idx % (BN / VEC)) * VEC; \
        int g_row  = k_base + k; \
        int g_col  = col_start + n_base; \
        float4 tmp = {0.f, 0.f, 0.f, 0.f}; \
        if (g_row < K && g_col + VEC - 1 < N) { \
            tmp = *reinterpret_cast<const float4*>(&B[g_row * N + g_col]); \
        } else if (g_row < K) { \
            _Pragma("unroll") \
            for (int v = 0; v < VEC; v++) \
                if (g_col + v < N) \
                    ((float*)&tmp)[v] = B[g_row * N + g_col + v]; \
        } \
        smem_B[(buf_idx)][k][n_base + 0] = tmp.x; \
        smem_B[(buf_idx)][k][n_base + 1] = tmp.y; \
        smem_B[(buf_idx)][k][n_base + 2] = tmp.z; \
        smem_B[(buf_idx)][k][n_base + 3] = tmp.w; \
    } \
}

template<
    int BM,    // = 128
    int BN,    // = 128
    int BK,    // = 16
    int TM,    // = 8
    int TN,    // = 8
    int PAD    // = 4
>
__global__ void gemm_v4_vec_dbuf(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    constexpr int THREAD_M  = BM / TM;         
    constexpr int THREAD_N  = BN / TN;         
    constexpr int THREADS   = THREAD_M * THREAD_N; 

    constexpr int VEC       = 4;
    constexpr int A_VEC_PER_ROW = BM / VEC;    
    constexpr int A_F4_PER_THREAD = (BK * BM / VEC) / THREADS; 
    constexpr int B_F4_PER_THREAD = (BK * BN / VEC) / THREADS; 

    int tid = threadIdx.x;
    int ty  = tid / THREAD_N;   
    int tx  = tid % THREAD_N;   

    int row_start = blockIdx.y * BM;
    int col_start = blockIdx.x * BN;

    __shared__ float smem_A[2][BK][BM + PAD];
    __shared__ float smem_B[2][BK][BN + PAD];

    float reg_C[TM][TN] = {};
    float reg_A[TM];
    float reg_B[TN];

    int num_tiles = (K + BK - 1) / BK;

    LOAD_TILE_MACRO(0, 0);
    __syncthreads();

    #pragma unroll
    for (int tile_k = 0; tile_k < num_tiles; tile_k++) {
        int cur  =  tile_k      & 1;   
        int next = (tile_k + 1) & 1;   

        if (tile_k + 1 < num_tiles) {
            LOAD_TILE_MACRO(tile_k + 1, next);
        }

        #pragma unroll
        for (int k = 0; k < BK; k++) {
            #pragma unroll
            for (int m = 0; m < TM; m++)
                reg_A[m] = smem_A[cur][k][ty * TM + m];

            #pragma unroll
            for (int n = 0; n < TN; n++)
                reg_B[n] = smem_B[cur][k][tx * TN + n];

            #pragma unroll
            for (int m = 0; m < TM; m++)
                #pragma unroll
                for (int n = 0; n < TN; n++)
                    reg_C[m][n] += reg_A[m] * reg_B[n];
        }

        __syncthreads();
    }

    #pragma unroll
    for (int m = 0; m < TM; m++) {
        int g_row = row_start + ty * TM + m;
        if (g_row >= M) continue;
        #pragma unroll
        for (int n = 0; n < TN; n += VEC) {
            int g_col = col_start + tx * TN + n;
            if (g_col + VEC - 1 < N) {
                float4 out = {reg_C[m][n], reg_C[m][n+1],
                              reg_C[m][n+2], reg_C[m][n+3]};
                *reinterpret_cast<float4*>(&C[g_row * N + g_col]) = out;
            } else {
                #pragma unroll
                for (int v = 0; v < VEC; v++)
                    if (g_col + v < N)
                        C[g_row * N + g_col + v] = reg_C[m][n + v];
            }
        }
    }
}

void cpu_gemm(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++) {
        float s = 0;
        for (int k = 0; k < K; k++) s += A[i*K+k] * B[k*N+j];
        C[i*N+j] = s;
    }
}

int main() {
    constexpr int BM=128, BN=128, BK=16, TM=8, TN=8, PAD=4;
    constexpr int THREADS = (BM/TM) * (BN/TN);  

    const int M=256, N=256, K=256;
    float *h_A=new float[M*K], *h_B=new float[K*N];
    float *h_Cgpu=new float[M*N], *h_Ccpu=new float[M*N];
    srand(42);
    for (int i=0;i<M*K;i++) h_A[i]=(float)rand()/RAND_MAX-0.5f;
    for (int i=0;i<K*N;i++) h_B[i]=(float)rand()/RAND_MAX-0.5f;

    float *d_A,*d_B,*d_C;
    cudaMalloc(&d_A, M*K*sizeof(float));
    cudaMalloc(&d_B, K*N*sizeof(float));
    cudaMalloc(&d_C, M*N*sizeof(float));
    cudaMemcpy(d_A, h_A, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K*N*sizeof(float), cudaMemcpyHostToDevice);

    dim3 blk(THREADS);
    dim3 grd((N+BN-1)/BN, (M+BM-1)/BM);

    // 👑 铁血修正点：还原清爽的三连角括号 <<< 👑
    gemm_v3_reg_block<BM,BN,BK,TM,TN><<<grd,blk>>>(d_A,d_B,d_C,M,N,K);
    cudaDeviceSynchronize();
    cudaMemcpy(h_Cgpu, d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost);
    cpu_gemm(h_A, h_B, h_Ccpu, M, N, K);
    float max_err=0;
    for (int i=0;i<M*N;i++) max_err=fmaxf(max_err,fabsf(h_Cgpu[i]-h_Ccpu[i]));
    printf("🚀 V3 Correctness: max_err=%.2e %s\n", max_err, max_err<0.01f?"✅":"❌");

    gemm_v4_vec_dbuf<BM,BN,BK,TM,TN,PAD><<<grd,blk>>>(d_A,d_B,d_C,M,N,K);
    cudaDeviceSynchronize();
    cudaMemcpy(h_Cgpu, d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost);
    max_err=0;
    for (int i=0;i<M*N;i++) max_err=fmaxf(max_err,fabsf(h_Cgpu[i]-h_Ccpu[i]));
    printf("🚀 V4 Correctness: max_err=%.2e %s\n", max_err, max_err<0.01f?"✅":"❌");

    // ── Benchmark ──
    int sizes[] = {512, 1024, 2048, 4096};
    printf("\n%-6s  %-12s  %-12s  %-12s  %-8s\n",
           "Size","V2-Tiled","V3-RegBlk","V4-Vec+DB","T4-TFLOPS");
    printf("──────────────────────────────────────────────────────────\n");

    for (int sz : sizes) {
        float *dA,*dB,*dC;
        cudaMalloc(&dA,(long long)sz*sz*sizeof(float));
        cudaMalloc(&dB,(long long)sz*sz*sizeof(float));
        cudaMalloc(&dC,(long long)sz*sz*sizeof(float));
        cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
        int runs=20;
        dim3 g3((sz+BN-1)/BN,(sz+BM-1)/BM);

        auto bench = [&](auto fn) -> float {
            fn(); cudaDeviceSynchronize();          
            cudaEventRecord(t0);
            for (int r=0;r<runs;r++) fn();
            cudaEventRecord(t1); cudaEventSynchronize(t1);
            float ms; cudaEventElapsedTime(&ms,t0,t1);
            return ms/runs;
        };

        // 👑 铁血修正点：Benchmark 的 3 处调用同步还原为 <<< 👑
        float v2 = bench([&](){ gemm_v2_tiled<16,16,16><<<dim3((sz+15)/16,(sz+15)/16),dim3(16,16)>>>(dA,dB,dC,sz,sz,sz); });
        float v3 = bench([&](){ gemm_v3_reg_block<BM,BN,BK,TM,TN><<<g3,blk>>>(dA,dB,dC,sz,sz,sz); });
        float v4 = bench([&](){ gemm_v4_vec_dbuf<BM,BN,BK,TM,TN,PAD><<<g3,blk>>>(dA,dB,dC,sz,sz,sz); });

        float tflops = 2.f*sz*sz*sz / (v4 * 1e-3f) * 1e-12f;
        printf("%-6d  %-12.3f  %-12.3f  %-12.3f  %-8.2f (较V2加速 %.1fx)\n",
               sz, v2, v3, v4, tflops, v2/v4);

        cudaEventDestroy(t0); cudaEventDestroy(t1);
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }

    delete[] h_A; delete[] h_B; delete[] h_Cgpu; delete[] h_Ccpu;
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}