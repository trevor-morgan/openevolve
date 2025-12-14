"""
CUDA GEMM Evaluator for OpenEvolve

Compiles and benchmarks CUDA kernels on a remote DGX Spark (GB10).
Returns fitness based on achieved TFLOP/s relative to theoretical peak.

GB10 Specs (Blackwell):
- FP32 peak: ~52 TFLOPS
- FP16 peak: ~209 TFLOPS
- Compute capability: 12.1
"""

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

# Configuration - adjust for your setup
REMOTE_HOST = "trevor-morgan@sparky.local"
NVCC_PATH = "/usr/local/cuda-13.0/bin/nvcc"
REMOTE_WORK_DIR = "/tmp/openevolve_cuda"

# GB10 theoretical peak (TFLOPS)
PEAK_FP32_TFLOPS = 52.0
PEAK_FP16_TFLOPS = 209.0

# Benchmark parameters
MATRIX_SIZES = [
    (1024, 1024, 1024),  # Small - warmup
    (2048, 2048, 2048),  # Medium
    (4096, 4096, 4096),  # Large - main benchmark
    (8192, 8192, 8192),  # XL - stress test
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
    """Create the benchmark harness code that will be compiled with the kernel."""
    return """
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

// External kernel interface
extern "C" void run_gemm(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream);

void init_matrix(float* mat, int size, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < size; i++) {
        mat[i] = (float)(rand()) / RAND_MAX * 2.0f - 1.0f;
    }
}

bool verify_gemm(const float* A, const float* B, const float* C, int M, int N, int K) {
    // Verify a few random elements
    const int num_checks = 10;
    float max_error = 0.0f;

    for (int check = 0; check < num_checks; check++) {
        int i = rand() % M;
        int j = rand() % N;
        float expected = 0.0f;
        for (int k = 0; k < K; k++) {
            expected += A[i * K + k] * B[k * N + j];
        }
        float error = fabsf(C[i * N + j] - expected) / (fabsf(expected) + 1e-6f);
        if (error > max_error) max_error = error;
    }

    return max_error < 0.01f;  // 1% relative error tolerance
}

double benchmark_gemm(int M, int N, int K, int warmup_runs, int benchmark_runs, bool* correct) {
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // Allocate host memory
    float* h_A = (float*)malloc(size_A);
    float* h_B = (float*)malloc(size_B);
    float* h_C = (float*)malloc(size_C);

    init_matrix(h_A, M * K, 42);
    init_matrix(h_B, K * N, 123);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Warmup
    for (int i = 0; i < warmup_runs; i++) {
        run_gemm(d_A, d_B, d_C, M, N, K, stream);
    }
    cudaStreamSynchronize(stream);

    // Check for errors after warmup
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\\n", cudaGetErrorString(err));
        *correct = false;
        free(h_A); free(h_B); free(h_C);
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        cudaStreamDestroy(stream);
        return 0.0;
    }

    // Verify correctness
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
    *correct = verify_gemm(h_A, h_B, h_C, M, N, K);

    if (!*correct) {
        free(h_A); free(h_B); free(h_C);
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        cudaStreamDestroy(stream);
        return 0.0;
    }

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, stream);
    for (int i = 0; i < benchmark_runs; i++) {
        run_gemm(d_A, d_B, d_C, M, N, K, stream);
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    double avg_ms = ms / benchmark_runs;

    // Calculate TFLOPS: 2*M*N*K FLOPs for GEMM
    double flops = 2.0 * M * N * K;
    double tflops = (flops / (avg_ms / 1000.0)) / 1e12;

    // Cleanup
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);

    return tflops;
}

int main() {
    printf("{\\n");
    printf("  \\"benchmarks\\": [\\n");

    int sizes[][3] = {{1024, 1024, 1024}, {2048, 2048, 2048}, {4096, 4096, 4096}, {8192, 8192, 8192}};
    int num_sizes = 4;

    double total_tflops = 0.0;
    int valid_benchmarks = 0;
    bool all_correct = true;

    for (int s = 0; s < num_sizes; s++) {
        int M = sizes[s][0];
        int N = sizes[s][1];
        int K = sizes[s][2];

        bool correct;
        double tflops = benchmark_gemm(M, N, K, 3, 10, &correct);

        printf("    {\\"M\\": %d, \\"N\\": %d, \\"K\\": %d, \\"tflops\\": %.4f, \\"correct\\": %s}",
               M, N, K, tflops, correct ? "true" : "false");

        if (s < num_sizes - 1) printf(",");
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
    Evaluate a CUDA GEMM kernel.

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
        kernel_path = os.path.join(tmpdir, "kernel.cu")
        harness_path = os.path.join(tmpdir, "benchmark.cu")

        # Write kernel code
        with open(kernel_path, "w") as f:
            f.write(program)

        # Write benchmark harness
        with open(harness_path, "w") as f:
            f.write(create_benchmark_harness())

        # Copy files to remote
        remote_kernel = f"{REMOTE_WORK_DIR}/kernel.cu"
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

        # Compile on remote
        compile_cmd = (
            f"{NVCC_PATH} -O3 -arch=sm_121 "  # sm_121 for Blackwell GB10
            f"-o {remote_binary} "
            f"{remote_kernel} {remote_harness} "
            f"-lcudart 2>&1"
        )

        ret, stdout, stderr = run_remote(compile_cmd, timeout=60)
        artifacts["compile_stdout"] = stdout
        artifacts["compile_stderr"] = stderr

        if ret != 0:
            # Compilation failed - try to extract useful error info
            error_msg = stdout + stderr
            return {
                "score": 0.0,
                "metrics": {"error": "Compilation failed", "details": error_msg[:1000]},
                "artifacts": artifacts,
            }

        # Run benchmark
        ret, stdout, stderr = run_remote(f"{remote_binary}", timeout=300)
        artifacts["benchmark_stdout"] = stdout
        artifacts["benchmark_stderr"] = stderr

        if ret != 0:
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

        # Score = achieved / peak (capped at 1.0)
        # Use FP32 peak since we're using float
        efficiency = min(avg_tflops / PEAK_FP32_TFLOPS, 1.0)

        # Add bonus metrics
        metrics["efficiency_percent"] = efficiency * 100
        metrics["peak_fp32_tflops"] = PEAK_FP32_TFLOPS
        metrics["combined_score"] = efficiency  # Use efficiency as the score for evolution

        return {"score": efficiency, "metrics": metrics, "artifacts": artifacts}


# For testing locally
if __name__ == "__main__":
    # Test with path to initial program
    example_dir = Path(__file__).parent
    initial_program_path = str(example_dir / "initial_program.cu")

    print("Testing CUDA GEMM evaluator...")
    result = evaluate(initial_program_path)

    print(f"\nScore: {result['score']:.4f}")
    print(f"Metrics: {json.dumps(result['metrics'], indent=2)}")

    if result["artifacts"]:
        print(f"\nCompile output: {result['artifacts'].get('compile_stdout', '')[:500]}")
