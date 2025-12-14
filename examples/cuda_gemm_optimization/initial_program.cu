// CUDA GEMM Kernel - Matrix Multiplication C = A * B
// Target: NVIDIA GB10 (DGX Spark) - Blackwell architecture, compute capability 12.1
// Goal: Maximize TFLOP/s through evolutionary optimization

#include <cuda_runtime.h>
#include <stdio.h>

// Matrix dimensions (will be set by evaluator)
#ifndef M_SIZE
#define M_SIZE 4096
#endif
#ifndef N_SIZE
#define N_SIZE 4096
#endif
#ifndef K_SIZE
#define K_SIZE 4096
#endif

// EVOLVE-BLOCK-START
// Naive GEMM kernel - evolve this for maximum performance
// Consider: tiling, shared memory, register blocking, tensor cores, vectorized loads
__global__ void gemm_kernel(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C,
                            int M, int N, int K) {
    // Each thread computes one element of C
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Launch configuration
void launch_gemm(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream) {
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);
    gemm_kernel<<<gridDim, blockDim, 0, stream>>>(A, B, C, M, N, K);
}
// EVOLVE-BLOCK-END

// Verification and benchmarking interface (do not modify)
extern "C" {
    void run_gemm(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream) {
        launch_gemm(A, B, C, M, N, K, stream);
    }
}
