// Fused Self-Attention Kernel for ML Training
// Target: NVIDIA GB10 (DGX Spark) - Blackwell architecture, compute capability 12.1
// Goal: Maximize TFLOP/s for transformer attention using FP16/BF16

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdio.h>
#include <cmath>

using namespace nvcuda;

// Attention parameters (typical transformer configs)
#ifndef BATCH_SIZE
#define BATCH_SIZE 8
#endif
#ifndef SEQ_LEN
#define SEQ_LEN 2048
#endif
#ifndef HEAD_DIM
#define HEAD_DIM 64
#endif
#ifndef NUM_HEADS
#define NUM_HEADS 32
#endif

// EVOLVE-BLOCK-START
// Naive fused attention: Q @ K^T -> softmax -> @ V
// This is intentionally unoptimized - evolve for FlashAttention-style performance
//
// Key optimizations to discover:
// 1. Tiling to fit in shared memory (FlashAttention style)
// 2. Online softmax (compute softmax incrementally)
// 3. Tensor core usage (WMMA/MMA for FP16)
// 4. Memory coalescing for Q, K, V loads
// 5. Fused scaling (1/sqrt(d_k))
// 6. Causal masking efficiency

constexpr int BLOCK_SIZE = 256;
constexpr float SOFTMAX_SCALE = 0.125f;  // 1/sqrt(64)

__global__ void attention_kernel(
    const half* __restrict__ Q,      // [batch, num_heads, seq_len, head_dim]
    const half* __restrict__ K,      // [batch, num_heads, seq_len, head_dim]
    const half* __restrict__ V,      // [batch, num_heads, seq_len, head_dim]
    half* __restrict__ O,            // [batch, num_heads, seq_len, head_dim]
    int batch_size, int num_heads, int seq_len, int head_dim,
    bool causal
) {
    // Each block handles one (batch, head, query_pos) combination
    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int query_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (batch_idx >= batch_size || head_idx >= num_heads || query_idx >= seq_len) return;

    // Pointers for this batch/head
    int bhq_offset = ((batch_idx * num_heads + head_idx) * seq_len + query_idx) * head_dim;
    int bh_offset = (batch_idx * num_heads + head_idx) * seq_len * head_dim;

    // Load query vector into registers (one thread per dimension, assumes head_dim <= BLOCK_SIZE)
    float q_reg = 0.0f;
    if (tid < head_dim) {
        q_reg = __half2float(Q[bhq_offset + tid]);
    }

    // Compute attention scores and accumulate output
    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    float out_accum[HEAD_DIM / BLOCK_SIZE + 1] = {0.0f};

    // Simple loop over all key positions (no tiling yet)
    int max_key = causal ? (query_idx + 1) : seq_len;

    for (int key_idx = 0; key_idx < max_key; key_idx++) {
        // Compute Q @ K^T for this key position
        int k_offset = bh_offset + key_idx * head_dim;

        float score = 0.0f;
        if (tid < head_dim) {
            float k_val = __half2float(K[k_offset + tid]);
            score = q_reg * k_val;
        }

        // Reduce score across threads (warp reduction)
        for (int offset = 16; offset > 0; offset /= 2) {
            score += __shfl_down_sync(0xffffffff, score, offset);
        }

        // Thread 0 has the dot product
        __shared__ float shared_score;
        if (tid == 0) {
            shared_score = score * SOFTMAX_SCALE;
        }
        __syncthreads();
        score = shared_score;

        // Online softmax update
        float old_max = max_score;
        max_score = fmaxf(max_score, score);
        float exp_diff = expf(old_max - max_score);
        sum_exp = sum_exp * exp_diff + expf(score - max_score);

        // Update output accumulator with rescaling
        float weight = expf(score - max_score);
        if (tid < head_dim) {
            float v_val = __half2float(V[bh_offset + key_idx * head_dim + tid]);
            out_accum[0] = out_accum[0] * exp_diff + weight * v_val;
        }
    }

    // Normalize and write output
    if (tid < head_dim) {
        float result = out_accum[0] / sum_exp;
        O[bhq_offset + tid] = __float2half(result);
    }
}

void launch_attention(
    const half* Q, const half* K, const half* V, half* O,
    int batch_size, int num_heads, int seq_len, int head_dim,
    bool causal, cudaStream_t stream
) {
    dim3 grid(seq_len, num_heads, batch_size);
    dim3 block(BLOCK_SIZE);

    attention_kernel<<<grid, block, 0, stream>>>(
        Q, K, V, O, batch_size, num_heads, seq_len, head_dim, causal
    );
}
// EVOLVE-BLOCK-END

// External interface for benchmarking
extern "C" {
    void run_attention(
        const half* Q, const half* K, const half* V, half* O,
        int batch_size, int num_heads, int seq_len, int head_dim,
        bool causal, cudaStream_t stream
    ) {
        launch_attention(Q, K, V, O, batch_size, num_heads, seq_len, head_dim, causal, stream);
    }
}
