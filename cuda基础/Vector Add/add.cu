#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define CEIL(a,b) ((a + b -1) / (b))
// __global__ 修饰符：告诉编译器这是GPU kernel函数
// 注意：__global__函数没有返回值（必须是void）

__global__ void vector_add_kernel(
    const float* __restrict__ x,   // __restrict__：提示编译器指针不重叠，可优化
    const float* __restrict__ y,
    float* output,
    int n
){
    // ---- 填空1 ----
    // 计算当前线程的全局索引
    // 公式：blockIdx.x * blockDim.x + threadIdx.x
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    // ---- 填空2 ----
    // 边界检查：只有gid < n的线程才做计算
    // 超出范围的线程什么都不做（直接return）
    if(gid >= n) return;
    // ---- 填空3 ----
    // 读取数据（直接数组下标）
    float val_x = x[gid];
    float val_y = y[gid];
    output[gid] = val_x + val_y;
}


// CPU参考实现（用于验证正确性）
void cpu_vector_add(const float* x, const float* y, float* out, int n) {
    for (int i = 0; i < n; i++) out[i] = x[i] + y[i];
}
int main(){
    const int N = 1 << 20;  // 1M 元素
    const int BLOCK_SIZE = 256;  // 每个block 256个线程（常用值）
    int num_blocks = CEIL(N,BLOCK_SIZE);

    // Host内存分配
    float *h_x = new float[N];
    float *h_y = new float[N];
    float *h_out_gpu = new float[N];
    float *h_out_cpu = new float[N];
    // 初始化数据
    for (int i = 0; i < N; i++) {
        h_x[i] = (float)i / N;
        h_y[i] = (float)(N - i) / N;
    }

    // Device内存分配
    float *d_x, *d_y, *d_out;
    cudaMalloc(&d_x,   N * sizeof(float));
    cudaMalloc(&d_y,   N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));

    // 数据从CPU拷到GPU
    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice);

    // ---- 填空6 ----
    // 启动kernel
    // 语法：kernel_name<<<num_blocks, BLOCK_SIZE>>>(参数...)
    vector_add_kernel<<<num_blocks, BLOCK_SIZE>>>(d_x, d_y, d_out, N);
    cudaDeviceSynchronize();

    // 把GPU结果拷回CPU
    cudaMemcpy(h_out_gpu, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);


    // CPU参考计算
    cpu_vector_add(h_x, h_y, h_out_cpu, N);


    // 验证正确性
    float max_diff = 0;
    for (int i = 0; i < N; i++) {
        max_diff = fmax(max_diff, fabs(h_out_gpu[i] - h_out_cpu[i]));
    }
    printf("Max diff: %.2e %s\n", max_diff, max_diff < 1e-5 ? "✅" : "❌");

    // 清理
    delete[] h_x; delete[] h_y;
    delete[] h_out_gpu; delete[] h_out_cpu;
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_out);

    return 0;
}