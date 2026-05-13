# PTXAS Optimization Example

Minimize register spills in PTX assembly with CompileIQ.

## Quick Start

```bash
python ptx_spill.py

# Custom architecture
python ptx_spill.py --arch sm_90a
```

## Requirements

- CUDA 13.3+
- Hopper GPU (sm_90a) or adjust `--arch`
- `pip install compileiq`

## How It Works

1. CompileIQ generates PTXAS control configurations
2. Each config compiles `w8_spill.ptx` with `--apply-controls`
3. Spill bytes are extracted from compiler output
4. Evolutionary search minimizes spills

## Files

- `ptx_spill.py` - Optimization script
- `w8_spill.ptx` - PTX file with register pressure
- `best_compiler_controls.acf` - Best config (generated)
