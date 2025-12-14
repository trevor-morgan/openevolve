# CUDA Fused Attention Optimization with OpenEvolve

Evolve FlashAttention-style fused attention kernels for maximum TFLOP/s on NVIDIA GB10 (DGX Spark).

## Why Attention?

For transformer-based ML training, attention is often the bottleneck:
- O(n^2) memory for naive implementation
- Multiple kernel launches (Q@K, softmax, @V)
- Memory bandwidth limited

FlashAttention showed 2-4x speedups by:
- Tiling to avoid materializing full attention matrix
- Online softmax computation
- Fusing all operations into one kernel

## Target Hardware

- **GPU**: NVIDIA GB10 (Blackwell architecture)
- **Compute Capability**: 12.1
- **FP16 Tensor Core Peak**: 209 TFLOPS
- **CUDA Version**: 13.0

## Benchmark Configurations

| Config | Batch | Heads | Seq Len | Head Dim | Use Case |
|--------|-------|-------|---------|----------|----------|
| Small | 8 | 32 | 512 | 64 | GPT-2 style |
| Medium | 4 | 32 | 1024 | 64 | Typical training |
| Large | 2 | 32 | 2048 | 64 | Long context |
| XL | 1 | 32 | 4096 | 64 | Stress test |

## Running

```bash
# Test evaluator standalone
python examples/cuda_attention_optimization/evaluator.py

# Run evolution
python openevolve-run.py \
    examples/cuda_attention_optimization/initial_program.cu \
    examples/cuda_attention_optimization/evaluator.py \
    --config examples/cuda_attention_optimization/config.yaml \
    --iterations 100
```

## Optimization Techniques to Discover

1. **Tiling** - Process attention in blocks that fit in shared memory
2. **Online Softmax** - Compute softmax incrementally, rescaling outputs
3. **Tensor Cores** - Use WMMA/MMA for FP16 matrix multiply
4. **Memory Coalescing** - Aligned, coalesced loads for Q, K, V
5. **Double Buffering** - Overlap compute with memory transfers
6. **Causal Masking** - Efficient masking without branches

## Expected Results

| Technique | Expected Efficiency |
|-----------|-------------------|
| Naive (initial) | 1-5% |
| Basic tiling | 10-20% |
| + Online softmax | 20-30% |
| + Tensor cores | 40-60% |
| FlashAttention-level | 60-80% |

## FLOPS Calculation

Attention FLOPS = 4 * batch * heads * seq_len^2 * head_dim

- Q @ K^T: 2 * batch * heads * seq^2 * d
- Attn @ V: 2 * batch * heads * seq^2 * d

For seq_len=2048, head_dim=64, batch=2, heads=32:
- FLOPS per forward = 4 * 2 * 32 * 2048^2 * 64 = 137 GFLOPS

## Files

- `initial_program.cu` - Naive attention kernel to evolve
- `evaluator.py` - Remote compilation and benchmarking
- `config.yaml` - OpenEvolve configuration
