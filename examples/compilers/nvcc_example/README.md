# NVCC Optimization Example

Optimize a CUDA reduction kernel with CompileIQ's evolutionary search.

## Quick Start

```bash
# Run optimization (auto-downloads search space)
python optimize_reduction.py

# Benchmark with optimized config
python optimize_reduction.py --benchmark-only --nvcc-options "--apply-controls reduction_best_config.bin"
```

## Options

```bash
# Custom GPU architecture
python optimize_reduction.py --arch sm_90a

# More thorough search
python optimize_reduction.py --generations 20 --pool-size 30

# Standalone benchmark (no optimization)
python optimize_reduction.py --benchmark-only --arch sm_100 --runs 5
```

## Requirements

- CUDA 13.3+
- Blackwell GPU (sm_100) or adjust `--arch`
- `pip install compileiq`

## Files

- `reduction.cu` - Self-contained CUDA reduction benchmark (single file)
- `optimize_reduction.py` - Optimization and benchmarking script
