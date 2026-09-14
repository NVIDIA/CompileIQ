# XLA-CIQ Experimental Reference Example

This directory contains experimental reference code for an XLA-side producer
flow. It uses CompileIQ to search PTXAS Advanced Control Files (ACFs) and
records selected ACF bytes in a protobuf bundle.

It illustrates one way an integration might parse XLA artifacts, invoke
CompileIQ, and assemble producer-side output. It is not a supported XLA
integration or an interoperable producer-consumer contract.

## Scope and boundaries

This example includes:

- `python -m xla_ciq.cli bundle populate`, which invokes either a deterministic
  mock or the CompileIQ search adapter and writes an experimental bundle;
- `python -m xla_ciq.cli bundle inspect`, which summarizes a bundle;
- `python -m xla_ciq.cli bundle evaluate`, which can re-measure bundled ACFs;
- `python -m xla_ciq.cli xla inspect`, which inspects XLA dump files; and
- an experimental fingerprint function and protobuf schema.

The downstream XLA consumer is not included. The fingerprint, routing map, and
bundle are illustrative producer-side structures. They intentionally do not
define how XLA would select or consume an ACF, what fallback behavior it would
use, or how compatibility would be maintained. They are not stable
cross-project contracts.

`xla_ciq.search.compileiq_runtime` implements the example's search interface
through the parent CompileIQ checkout. CompileIQ supplies search primitives;
it does not define the XLA artifact or consumer behavior described above.
See [`workflow.md`](workflow.md) for the producer flow and failure behavior.

## Requirements

- Python 3.11, 3.12, or 3.13
- The parent CompileIQ checkout and its normal runtime dependencies
- Click and protobuf for mock and inspection commands
- CUDA Python, a CUDA Toolkit installation, and a compatible NVIDIA GPU for
  illustrative real timing

The deterministic mock path does not require CompileIQ or a GPU.

## Set up the example

```bash
# From examples/compilers/xla_ciq_example
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ../../..
python -m pip install -r requirements-dev.txt
export PYTHONPATH="$PWD/src"
```

For the real timing path, also install `requirements-real.txt`. This source
layout is run in place; there is no wheel or installed `xla-ciq` executable.

## Quick start with the mock adapter

Mock mode is the default. Passing `--mock` below makes that choice explicit.

```bash
python -m xla_ciq.cli xla inspect tests/fixtures/xla_dump --json

python -m xla_ciq.cli bundle populate ./kernels.xla-ciq.bundle \
  --kernel-name fusion_demo \
  --input tests/fixtures/sample.ptx \
  --shapes 'f32[1024],f32[2048]' \
  --mock

python -m xla_ciq.cli bundle inspect ./kernels.xla-ciq.bundle --json

python -m xla_ciq.cli bundle evaluate ./kernels.xla-ciq.bundle \
  --input tests/fixtures/sample.ptx \
  --kernel-name fusion_demo \
  --mock \
  --json
```

## Illustrative real timing

Real execution must be selected explicitly with `--real`. It requires
CompileIQ, `ptxas`, a compatible NVIDIA GPU, and launch metadata sufficient to
invoke the selected kernel.

```bash
python -m xla_ciq.cli bundle populate ./kernels.xla-ciq.bundle \
  --kernel-name <kernel> \
  --input <kernel.ptx> \
  --shapes 'f32[1024]' \
  --dump-dir <xla-dump-directory> \
  --real \
  --task-timeout 30
```

The real path is a synthetic timing probe. It creates zero-filled, independent
buffers for pointer parameters and does not compare kernel outputs with
expected results. Scalar value parameters are rejected. Do not treat a
generated ACF as correctness-validated or production-ready. Every CompileIQ
candidate uses an isolated worker so a timeout or CUDA fault cannot contaminate
a later candidate.

`--task-timeout` is CompileIQ's per-objective timeout. It is not an overall
search deadline.

Tensor Memory Accelerator (TMA) tile sweeping is retained as an experiment.
The bundle does not serialize a winning TMA box or swizzle that differs from
the seed layout. If a search selects such a winner, the producer fails before
saving the bundle and reports both the seed and winning settings. A
seed-equivalent winner may be serialized.

Review all command options with:

```bash
python -m xla_ciq.cli --help
python -m xla_ciq.cli bundle populate --help
python -m xla_ciq.cli bundle inspect --help
python -m xla_ciq.cli bundle evaluate --help
python -m xla_ciq.cli xla inspect --help
```

## Bundle schema

The source schema is `schemas/xla_ciq_bundle.proto`. Serialization and
producer-side routing are retained because they are part of the example, not
because the schema is a supported external interface:

```proto
package xla_ciq.bundle.v1alpha1;

message XlaCiqTuningBundle {
  uint32 format_version = 1;
  map<string, string> fingerprint_to_control_file_id = 2;
  map<string, AdvancedControlFile> control_files = 3;
  map<string, FingerprintMetadata> fingerprint_to_metadata = 4;
}
```

The checked-in Python binding is generated with `grpcio-tools==1.83.1`:

```bash
./scripts/generate_proto.sh
```

The reader rejects unsupported versions, incorrect content hashes, missing
routes, and metadata without a route. Fingerprint recomputation must also match
before an ACF is selected.

## Validation

```bash
PYTHONPATH=src python -m ruff check src tests
PYTHONPATH=src pytest -q
PYTHONPATH=src python -m pytest -q
./scripts/generate_proto.sh
git diff --exit-code
```

The parent repository CI runs the mock tests and checks that the protobuf
binding is reproducible. These deterministic checks do not constitute real
CompileIQ, CUDA, GPU, output-correctness, or downstream XLA-consumer validation.

## License

This example is covered by the repository's root `LICENSE`. Dependencies retain
their respective licenses.
