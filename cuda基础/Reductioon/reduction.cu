// reduction.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

// 第一版：最直接的实现
// 每个block把自己负责的BLOCK_SIZE个元素求和，结果写到output[blockIdx.x]
// 主程序再把各block结果求和（两级reduction）
template<typename BLOCK_SIZE>
__global__ void reduce_sum_v1(
    const float* __restrict__ input,
    float* output,
    int N
){
    // ---- 填空1 ----
    // 声明shared memory
    // 大小固定为BLOCK_SIZE个float（BLOCK_SIZE是模板参数，编译期已知）
    __shared__ float smem[BLOCK_SIZE];
    // ---- 填空2 ----
    // 计算线程的局部id（block内）和全局id
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // ---- 填空3 ----
    // 加载数据到shared memory
    // 越界的线程：smem[tid] = 0.0（不影响求和）

    smem[tid] = (gid < N) ? input[gid] : 0.0f;


    // ---- 填空4 ----
    // 等block内所有线程完成加载
    // （不加这句：某些线程可能读到其他线程还没写入的smem值）
    __syncthreads();


    // ---- 填空5 ----
    // Tree Reduction主循环
    // stride从BLOCK_SIZE/2开始，每轮减半
    for(int stride = BLOCK_SIZE / 2;stride > 0;stride >>= 1){
        if(tid < stride){
            smem[tid] += smem[tid + stride];
        }
        // ---- 填空6 ----
        // 每轮结束后同步（确保下一轮读到的是本轮写入的结果）
        __syncthreads();
    }
    if(tid == 0){
        output[blockIdx.x] = smem[0];     //块号
    }
}

int main(){
    const int N = 1 << 24;        // 16M元素

    const int BLOCK_SIZE = 256;

    const int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float *h_in  = new float[N];
    float *h_out = new float[num_blocks];
    float *d_in, *d_out;

    // 初始化：全1，期望sum = N
    for (int i = 0; i < N; i++) h_in[i] = 1.0f;


    cudaMalloc(&d_in,  N          * sizeof(float));
    cudaMalloc(&d_out, num_blocks  * sizeof(float));
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    reduce_sum_v1<BLOCK_SIZE><<<num_blocks, BLOCK_SIZE>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    // 把block结果拷回，CPU做最终求和
    cudaMemcpy(h_out, d_out, num_blocks * sizeof(float), cudaMemcpyDeviceToHost);

    float gpu_sum = 0;
    for (int i = 0; i < num_blocks; i++) gpu_sum += h_out[i];

    printf("GPU sum: %.0f\n", gpu_sum);
    printf("Expected:%.0f\n", (float)N);
    printf("Result: %s\n", fabs(gpu_sum - N) < 1.0f ? "✅" : "❌");

    delete[] h_in; delete[] h_out;
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}