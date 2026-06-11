# Triton Optimization Examples

Optimize Triton kernels with CompileIQ's PTXAS controls.

## Examples

### Basic PTX Controls (`triton_ptx.py`)

Optimizes a fixed-config matmul kernel by tuning PTXAS compiler settings.

```bash
python triton_ptx.py
```

### Mixed Search Space (`mixed_triton.py`)

Searches both Triton configs AND PTXAS controls together:
- **User space**: Block sizes, warps, stages
- **Compiler space**: PTXAS compiler controls

```bash
python mixed_triton.py
```

## Requirements

- CUDA 13.3+
- PyTorch with CUDA
- Triton (`pip install triton`)
- `pip install compileiq`

## Output

Both scripts generate `best_matmul.acf` - use with Triton's `ptx_options` or through `PTXAS_OPTIONS` env var:

```python
kernel[grid](..., ptx_options="--apply-controls=best_matmul.acf")
```
