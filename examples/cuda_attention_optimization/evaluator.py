"""
CUDA Fused Attention Evaluator for OpenEvolve

Compiles and benchmarks attention kernels on a remote DGX Spark (GB10).
Returns fitness based on achieved TFLOP/s relative to theoretical peak.

GB10 Specs (Blackwell):
- FP16 peak: ~209 TFLOPS (tensor cores)
- BF16 peak: ~209 TFLOPS
- TF32 peak: ~104 TFLOPS

Attention FLOPS calculation:
- Q @ K^T: 2 * batch * heads * seq_len^2 * head_dim
- Softmax: ~5 * batch * heads * seq_len^2 (exp, sum, div)
- Attn @ V: 2 * batch * heads * seq_len^2 * head_dim
- Total: ~4 * batch * heads * seq_len^2 * head_dim (dominated by matmuls)
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path

# Configuration - adjust for your setup
REMOTE_HOST = "trevor-morgan@sparky.local"
NVCC_PATH = "/usr/local/cuda-13.0/bin/nvcc"
REMOTE_WORK_DIR = "/tmp/openevolve_attention"

# GB10 theoretical peak (TFLOPS) for FP16 tensor cores
PEAK_FP16_TFLOPS = 209.0

# Benchmark configurations: (batch_size, num_heads, seq_len, head_dim)
BENCHMARK_CONFIGS = [
    (8, 32, 512, 64),  # Small - GPT-2 style
    (4, 32, 1024, 64),  # Medium - typical training
    (2, 32, 2048, 64),  # Large - long context
    (1, 32, 4096, 64),  # XL - stress test
]
WARMUP_RUNS = 3
BENCHMARK_RUNS = 10


def run_remote(cmd: str, timeout: int = 120) -> tuple[int, str, str]:
    """Run command on remote host via SSH."""
    full_cmd = f'ssh {REMOTE_HOST} "{cmd}"'
    result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return result.returncode, result.stdout, result.stderr


def copy_to_remote(local_path: str, remote_path: str) -> bool:
    """Copy file to remote host via SCP."""
    cmd = f"scp {local_path} {REMOTE_HOST}:{remote_path}"
    result = subprocess.run(cmd, shell=True, capture_output=True, timeout=30)
    return result.returncode == 0


def create_benchmark_harness() -> str:
    """Create the benchmark harness code for attention kernel."""
    return """
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cmath>

// External kernel interface
extern "C" void run_attention(
    const half* Q, const half* K, const half* V, half* O,
    int batch_size, int num_heads, int seq_len, int head_dim,
    bool causal, cudaStream_t stream
);

void init_matrix_half(half* mat, int size, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < size; i++) {
        // Small random values to avoid overflow in softmax
        float val = ((float)(rand()) / RAND_MAX - 0.5f) * 0.1f;
        mat[i] = __float2half(val);
    }
}

// Simple reference attention for verification (CPU, very slow)
bool verify_attention_simple(
    const half* Q, const half* K, const half* V, const half* O,
    int batch_size, int num_heads, int seq_len, int head_dim,
    bool causal
) {
    // Only verify a few random positions to avoid slow CPU computation
    const int num_checks = 5;
    float max_error = 0.0f;
    float scale = 1.0f / sqrtf((float)head_dim);

    srand(12345);
    for (int check = 0; check < num_checks; check++) {
        int b = rand() % batch_size;
        int h = rand() % num_heads;
        int q_pos = rand() % seq_len;
        int d = rand() % head_dim;

        // Compute reference attention for this position
        int max_k = causal ? (q_pos + 1) : seq_len;
        int qkv_offset = ((b * num_heads + h) * seq_len) * head_dim;

        // Compute scores and softmax
        float max_score = -INFINITY;
        float* scores = (float*)malloc(max_k * sizeof(float));

        for (int k_pos = 0; k_pos < max_k; k_pos++) {
            float score = 0.0f;
            for (int dd = 0; dd < head_dim; dd++) {
                float q_val = __half2float(Q[qkv_offset + q_pos * head_dim + dd]);
                float k_val = __half2float(K[qkv_offset + k_pos * head_dim + dd]);
                score += q_val * k_val;
            }
            scores[k_pos] = score * scale;
            max_score = fmaxf(max_score, scores[k_pos]);
        }

        float sum_exp = 0.0f;
        for (int k_pos = 0; k_pos < max_k; k_pos++) {
            scores[k_pos] = expf(scores[k_pos] - max_score);
            sum_exp += scores[k_pos];
        }

        // Compute output for dimension d
        float expected = 0.0f;
        for (int k_pos = 0; k_pos < max_k; k_pos++) {
            float weight = scores[k_pos] / sum_exp;
            float v_val = __half2float(V[qkv_offset + k_pos * head_dim + d]);
            expected += weight * v_val;
        }

        float actual = __half2float(O[qkv_offset + q_pos * head_dim + d]);
        float error = fabsf(actual - expected) / (fabsf(expected) + 1e-5f);
        if (error > max_error) max_error = error;

        free(scores);
    }

    // FP16 tolerance is higher due to precision loss
    return max_error < 0.05f;  // 5% relative error tolerance for FP16
}

double benchmark_attention(
    int batch_size, int num_heads, int seq_len, int head_dim,
    bool causal, int warmup_runs, int benchmark_runs, bool* correct
) {
    size_t qkv_size = batch_size * num_heads * seq_len * head_dim * sizeof(half);

    // Allocate host memory
    half* h_Q = (half*)malloc(qkv_size);
    half* h_K = (half*)malloc(qkv_size);
    half* h_V = (half*)malloc(qkv_size);
    half* h_O = (half*)malloc(qkv_size);

    init_matrix_half(h_Q, batch_size * num_heads * seq_len * head_dim, 42);
    init_matrix_half(h_K, batch_size * num_heads * seq_len * head_dim, 123);
    init_matrix_half(h_V, batch_size * num_heads * seq_len * head_dim, 456);

    // Allocate device memory
    half *d_Q, *d_K, *d_V, *d_O;
    cudaMalloc(&d_Q, qkv_size);
    cudaMalloc(&d_K, qkv_size);
    cudaMalloc(&d_V, qkv_size);
    cudaMalloc(&d_O, qkv_size);

    cudaMemcpy(d_Q, h_Q, qkv_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, qkv_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, qkv_size, cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Warmup
    for (int i = 0; i < warmup_runs; i++) {
        run_attention(d_Q, d_K, d_V, d_O, batch_size, num_heads, seq_len, head_dim, causal, stream);
    }
    cudaStreamSynchronize(stream);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\\n", cudaGetErrorString(err));
        *correct = false;
        free(h_Q); free(h_K); free(h_V); free(h_O);
        cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
        cudaStreamDestroy(stream);
        return 0.0;
    }

    // Verify correctness
    cudaMemcpy(h_O, d_O, qkv_size, cudaMemcpyDeviceToHost);
    *correct = verify_attention_simple(h_Q, h_K, h_V, h_O, batch_size, num_heads, seq_len, head_dim, causal);

    if (!*correct) {
        free(h_Q); free(h_K); free(h_V); free(h_O);
        cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
        cudaStreamDestroy(stream);
        return 0.0;
    }

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, stream);
    for (int i = 0; i < benchmark_runs; i++) {
        run_attention(d_Q, d_K, d_V, d_O, batch_size, num_heads, seq_len, head_dim, causal, stream);
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    double avg_ms = ms / benchmark_runs;

    // Calculate TFLOPS
    // Attention FLOPS: 4 * batch * heads * seq^2 * head_dim (two matmuls)
    double flops = 4.0 * batch_size * num_heads * (double)seq_len * seq_len * head_dim;
    double tflops = (flops / (avg_ms / 1000.0)) / 1e12;

    // Cleanup
    free(h_Q); free(h_K); free(h_V); free(h_O);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);

    return tflops;
}

int main() {
    printf("{\\n");
    printf("  \\"benchmarks\\": [\\n");

    // Benchmark configs: batch, heads, seq_len, head_dim
    int configs[][4] = {{8, 32, 512, 64}, {4, 32, 1024, 64}, {2, 32, 2048, 64}, {1, 32, 4096, 64}};
    int num_configs = 4;

    double total_tflops = 0.0;
    int valid_benchmarks = 0;
    bool all_correct = true;

    for (int c = 0; c < num_configs; c++) {
        int batch = configs[c][0];
        int heads = configs[c][1];
        int seq_len = configs[c][2];
        int head_dim = configs[c][3];

        bool correct;
        double tflops = benchmark_attention(batch, heads, seq_len, head_dim, true, 3, 10, &correct);

        printf("    {\\"batch\\": %d, \\"heads\\": %d, \\"seq_len\\": %d, \\"head_dim\\": %d, \\"tflops\\": %.4f, \\"correct\\": %s}",
               batch, heads, seq_len, head_dim, tflops, correct ? "true" : "false");

        if (c < num_configs - 1) printf(",");
        printf("\\n");

        if (correct && tflops > 0) {
            total_tflops += tflops;
            valid_benchmarks++;
        }
        if (!correct) all_correct = false;
    }

    double avg_tflops = valid_benchmarks > 0 ? total_tflops / valid_benchmarks : 0.0;

    printf("  ],\\n");
    printf("  \\"avg_tflops\\": %.4f,\\n", avg_tflops);
    printf("  \\"all_correct\\": %s,\\n", all_correct ? "true" : "false");
    printf("  \\"valid_benchmarks\\": %d\\n", valid_benchmarks);
    printf("}\\n");

    return all_correct ? 0 : 1;
}
"""


def evaluate(program_path: str) -> dict:
    """
    Evaluate a CUDA attention kernel.

    Args:
        program_path: Path to the CUDA kernel file to evaluate

    Returns:
        dict with 'score' (0.0-1.0 based on TFLOP/s vs peak),
        'metrics' (detailed benchmark results),
        and 'artifacts' (compilation logs, etc.)
    """
    artifacts = {}
    metrics = {}

    # Read the program code from the file path
    with open(program_path) as f:
        program = f.read()

    # Setup remote directory
    run_remote(f"mkdir -p {REMOTE_WORK_DIR}")

    # Create temporary files locally
    with tempfile.TemporaryDirectory() as tmpdir:
        kernel_path = os.path.join(tmpdir, "attention.cu")
        harness_path = os.path.join(tmpdir, "benchmark.cu")

        # Write kernel code
        with open(kernel_path, "w") as f:
            f.write(program)

        # Write benchmark harness
        with open(harness_path, "w") as f:
            f.write(create_benchmark_harness())

        # Copy files to remote
        remote_kernel = f"{REMOTE_WORK_DIR}/attention.cu"
        remote_harness = f"{REMOTE_WORK_DIR}/benchmark.cu"
        remote_binary = f"{REMOTE_WORK_DIR}/benchmark"

        if not copy_to_remote(kernel_path, remote_kernel):
            return {
                "score": 0.0,
                "metrics": {"error": "Failed to copy kernel to remote"},
                "artifacts": artifacts,
            }

        if not copy_to_remote(harness_path, remote_harness):
            return {
                "score": 0.0,
                "metrics": {"error": "Failed to copy harness to remote"},
                "artifacts": artifacts,
            }

        # Compile on remote - need FP16 support
        compile_cmd = (
            f"{NVCC_PATH} -O3 -arch=sm_121 "  # sm_121 for Blackwell GB10
            f"--use_fast_math "
            f"-o {remote_binary} "
            f"{remote_kernel} {remote_harness} "
            f"-lcudart 2>&1"
        )

        ret, stdout, stderr = run_remote(compile_cmd, timeout=120)
        artifacts["compile_stdout"] = stdout
        artifacts["compile_stderr"] = stderr

        if ret != 0:
            error_msg = stdout + stderr
            return {
                "score": 0.0,
                "metrics": {"error": "Compilation failed", "details": error_msg[:1000]},
                "artifacts": artifacts,
            }

        # Run benchmark
        ret, stdout, stderr = run_remote(f"{remote_binary}", timeout=600)
        artifacts["benchmark_stdout"] = stdout
        artifacts["benchmark_stderr"] = stderr

        if ret != 0 and "all_correct" not in stdout:
            return {
                "score": 0.0,
                "metrics": {"error": "Benchmark failed", "details": stderr[:1000]},
                "artifacts": artifacts,
            }

        # Parse JSON results
        try:
            results = json.loads(stdout)
        except json.JSONDecodeError as e:
            return {
                "score": 0.0,
                "metrics": {"error": f"Failed to parse results: {e}", "raw_output": stdout[:1000]},
                "artifacts": artifacts,
            }

        metrics = results

        # Calculate score based on achieved vs peak TFLOP/s
        if not results.get("all_correct", False):
            return {
                "score": 0.0,
                "metrics": {**metrics, "error": "Correctness check failed"},
                "artifacts": artifacts,
            }

        avg_tflops = results.get("avg_tflops", 0.0)

        # Score = achieved / peak (FP16 tensor core peak)
        efficiency = min(avg_tflops / PEAK_FP16_TFLOPS, 1.0)

        metrics["efficiency_percent"] = efficiency * 100
        metrics["peak_fp16_tflops"] = PEAK_FP16_TFLOPS
        metrics["combined_score"] = efficiency  # Use efficiency as the score for evolution

        return {"score": efficiency, "metrics": metrics, "artifacts": artifacts}


# For testing locally
if __name__ == "__main__":
    example_dir = Path(__file__).parent
    initial_program_path = str(example_dir / "initial_program.cu")

    print("Testing CUDA Attention evaluator...")
    result = evaluate(initial_program_path)

    print(f"\nScore: {result['score']:.4f}")
    print(f"Metrics: {json.dumps(result['metrics'], indent=2)}")

    if result["artifacts"]:
        print(f"\nCompile output: {result['artifacts'].get('compile_stdout', '')[:500]}")
