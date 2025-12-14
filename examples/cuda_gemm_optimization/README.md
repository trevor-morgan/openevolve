# CUDA GEMM Optimization with OpenEvolve

Evolve CUDA matrix multiplication kernels for maximum TFLOP/s on NVIDIA GB10 (DGX Spark).

## Target Hardware

- **GPU**: NVIDIA GB10 (Blackwell architecture)
- **Compute Capability**: 12.1
- **Theoretical Peak**: 52 TFLOP/s (FP32), 209 TFLOP/s (FP16)
- **CUDA Version**: 13.0

## Setup

1. Ensure SSH access to your DGX Spark:
   ```bash
   ssh trevor-morgan@sparky.local nvidia-smi
   ```

2. Update `evaluator.py` with your remote host if different:
   ```python
   REMOTE_HOST = "trevor-morgan@sparky.local"
   ```

3. Start CLIProxyAPI for LLM access:
   ```bash
   cd ~/cliproxyapi && ./cli-proxy-api
   ```

## Running

```bash
cd /path/to/openevolve

# Run evolution
python openevolve-run.py \
    examples/cuda_gemm_optimization/initial_program.cu \
    examples/cuda_gemm_optimization/evaluator.py \
    --config examples/cuda_gemm_optimization/config.yaml \
    --iterations 100

# Resume from checkpoint
python openevolve-run.py \
    examples/cuda_gemm_optimization/initial_program.cu \
    examples/cuda_gemm_optimization/evaluator.py \
    --config examples/cuda_gemm_optimization/config.yaml \
    --checkpoint examples/cuda_gemm_optimization/openevolve_output/checkpoints/checkpoint_50 \
    --iterations 50
```

## Test Evaluator Standalone

```bash
python examples/cuda_gemm_optimization/evaluator.py
```

## Optimization Techniques

The LLM will explore:

1. **Tiling with shared memory** - Reduce global memory bandwidth
2. **Register blocking** - Compute NxM output tile per thread
3. **Vectorized loads** - float4 for coalesced 128-bit loads
4. **Tensor cores** - WMMA/MMA for FP16/TF32 acceleration
5. **Double buffering** - Overlap compute with memory loads
6. **Bank conflict avoidance** - Pad shared memory arrays
7. **Loop unrolling** - Eliminate loop overhead

## Expected Results

| Technique | Expected Efficiency |
|-----------|-------------------|
| Naive (initial) | 1-5% |
| Basic tiling | 10-20% |
| Register blocking | 30-50% |
| Optimized tiling + vectorized | 50-70% |
| Tensor core (FP16) | 70-90% |
| cuBLAS-level | 85-95% |

## Metrics

The evaluator reports:
- `avg_tflops`: Average TFLOP/s across matrix sizes
- `efficiency_percent`: Achieved vs theoretical peak
- `benchmarks`: Per-size breakdown (1K, 2K, 4K, 8K matrices)
- `all_correct`: Correctness verification passed

## Files

- `initial_program.cu` - Naive GEMM kernel to evolve
- `evaluator.py` - Remote compilation and benchmarking
- `config.yaml` - OpenEvolve configuration
