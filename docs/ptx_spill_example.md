# Tuning PTXAS Advanced Controls for a CUDA Kernel

In this section, we will walk you through how to set up your first search to tune PTXAS compiler controls.

> The example code and supporting files can be found [in our repo here](https://github.com/NVIDIA/CompileIQ/blob/main/examples/compilers/ptxas_example/ptx_spill.py).

## Register Spill Example

In this example, we will use a PTX kernel that contains register spill issues. Register spilling occurs when there aren’t enough GPU registers to hold active variables. These variables are temporarily moved to slower memory and then loaded back when needed, which can slow performance due to increased memory traffic.

Our end goal is to run a search that tunes PTXAS compiler controls and reduces register spilling to (or close to) zero.

What you’ll need:

* A Python environment with CompileIQ installed
* PTXAS 13.3 (or CUDA Toolkit (CTK) 13.3)

Because we are only compiling code, you do not need a GPU to run this example.

### Preparing the objective

Our first step is to define our objective function.

We need to call `ptxas` to compile the kernel and retrieve the spill cost. A simple way to do this is with Python’s `subprocess` module:

```python
PTX_SOURCE_PATH = "w8_spill.ptx"
ptxas_command = [
        "ptxas",
        "-v",
        "-arch=sm_90a",
        PTX_SOURCE_PATH,
    ]
ptxas_proc = subprocess.run(ptxas_command,text=True)
```

The `-v` option should produce verbose output like this:

```markdown
ptxas info    : 11 bytes gmem
ptxas info    : Compiling entry function '_attn_fwd' for 'sm_90a'
ptxas info    : Function properties for _attn_fwd
    56 bytes stack frame, 52 bytes spill stores, 52 bytes spill loads
ptxas info    : Used 240 registers, used 1 barriers, 56 bytes cumulative stack size
ptxas info    : Compile time = 372.065 ms
```

The line we are looking for is: `56 bytes stack frame, 52 bytes spill stores, 52 bytes spill loads`. Here, we see that 52 bytes were spilled. We can parse the output with a simple regex:

```python
spillBytes = float(re.findall(r"(\d+) bytes spill stores", ptxas_proc.stdout)[0])
```

Now we know how to compile our sample kernel and extract our metric. The only thing left is passing Advanced Controls File (ACF) sampled by CompileIQ to PTXAS. For that, a small change to the command line is enough:

```python

ptxas_command = [
    "ptxas",
    "-v",
    "-arch=sm_90a",
    "--apply-controls",
    "compiler_config.acf",
    PTX_SOURCE_PATH,
]

ptxas_proc = subprocess.run(ptxas_command, text=True)
```

The `--apply-controls` option is available in PTXAS and NVCC starting in CTK 13.3, and it expects an ACF file.

With these building blocks, we can now write the objective function:

```python
import subprocess
from uuid import uuid4
import re
import os
from compileiq.utils.helpers import save_compiler_config
from compileiq.types import INVALID_SCORE, SearchConfiguration
from compileiq.ciq import Search


PTX_SOURCE_PATH = "w8_spill.ptx"

def objective(config):
    """
    This function will receive a string blob that contains the compiler controls configuration.
    It will save it to a temporary file, and then call ptxas with that configuration to compile
    the PTX file. Finally, it will parse the output of ptxas to extract the number of register spills.
    """

    tmp_filename = f"tmp_{uuid4().hex}.acf"
    save_compiler_config(tmp_filename, config)

    ptxas_command = [
        "ptxas",
        "-v",
        "-arch=sm_90a",
        "--apply-controls",
        tmp_filename,
        PTX_SOURCE_PATH,
    ]
    try:
        ptxas_proc = subprocess.run(
            ptxas_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
            text=True,
        )

        if ptxas_proc.returncode != 0:
            raise RuntimeError(
                f"ptxas command failed with return code {ptxas_proc.returncode}. "
                f"Output: {ptxas_proc.stdout}"
            )

        # Parsing cmd output
        try:
            spillBytes = float(re.findall(r"(\d+) bytes spill stores", ptxas_proc.stdout)[0])
        except (ValueError, IndexError):
            logger.error("Error when parsing register spills from ptxas output.")
            raise

    except Exception as e:
        logger.exception(e)
        spillBytes = INVALID_SCORE
    finally:
        if os.path.exists(tmp_filename):
            os.remove(tmp_filename)

    return spillBytes
```

The `objective` will execute multiple times for different sampled data. It receives a string blob needs to be saved to a file. CompileIQ provides the function `save_compiler_config` to help you prepare the file. Our return score/metric that we want to optimize is the number of spilled bytes extracted through the regex.

The `objective` will execute multiple times for different sampled configurations. It receives a string blob that needs to be saved to a file. CompileIQ provides `save_compiler_config` to help with this. The score we optimize is the number of spilled bytes extracted via the regex.

Whenever the code errors out, we return `INVALID_SCORE` (the special `'*'` value). CompileIQ treats these as invalid evaluations, so returning `INVALID_SCORE` is preferred over something like `float('inf')`.

In this example, we create a temporary file to store the ACF and remove it afterward. Notice how much of the code is dedicated to error handling, as suggested in our [Safety Section](compilers_overview.md#safety--correctness-read-this-first). Ideally, you would also run the kernel and verify correctness, but we skip that here so the example can run on CPU-only systems.

### Configuring the Search

With the objective in place, we can now configure and start the search. The key point is to point to the PTXAS 13.3 search-space file so CompileIQ samples PTXAS ACFs.

```python
def main():
    main_config = SearchConfiguration(
        pool_size=32,
        generations=5,
        mutate_rate=0.3,
        problem_type="min",
        num_objectives=1,
    )

    cuda_version = re.search(r"release (\d+\.\d+),", version_output).group(1)

    tuner = Search(
        objective_function=objective,
        search_space=PtxasSearchSpace(version=cuda_version),
        search_config=main_config,
    )

    # Starting the tuning process with 4 parallel workers
    results = tuner.start(num_workers=4)

    best = results.get_best_result()
    logger.info(f"Best spill found: {best['score_1']}")
    save_compiler_config("best_compiler_controls.acf", best["params"])
```

The structure is mostly the same as other CompileIQ examples. Here we use the default worker (multiprocess), which executes 4 objective evaluations in parallel.

We save the best solution to `best_compiler_controls.acf`. You can call `ptxas` from the command line to reproduce the result:

```bash
ptxas -v -arch=sm_90a --apply-controls best_compiler_controls.acf w8_spill.ptx
```

Expected output:

```markdown
ptxas info    : 11 bytes gmem
ptxas info    : Compiling entry function '_attn_fwd' for 'sm_90a'
ptxas info    : Function properties for _attn_fwd
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 240 registers, used 1 barriers
ptxas info    : Compile time = 369.596 ms
```

## Other compilers and environments

Our current support extends to PTXAS and NVCC. This option offers the user to tune standalone NVCC, standalone PTXAS or a search space with both combined. 

Other environments like Triton and Helion provide facilities to inject ACFs on the compilation flow, and even to select ACFs for specific kernels or inputs. 

> **NOTE**: ACFs can contain controls for specific compilers. When using a PTXAS ACF with NVCC, pass it directly to PTXAS via:
>
>           nvcc -Xptxas="--apply-controls=best_config.bin" kernel.cu
>

Consult the respective documentations or follow our examples in the next pages of this guide:

* [Tuning NVCC compiler controls](nvcc_example.md)
* [Tuning PTXAS controls in your Triton kernel](triton_example.md)

