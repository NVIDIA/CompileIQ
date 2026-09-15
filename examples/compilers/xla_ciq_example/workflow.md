# XLA-CIQ Experimental Producer Workflow

This document describes an illustrative XLA-side producer flow. The example
calls CompileIQ through an adapter and owns only the interpretation and
serialization performed by this source tree.

## Responsibility map

| Concern | Scope in this example |
| --- | --- |
| Search algorithms and compiler search-space primitives | Parent CompileIQ package |
| Reading XLA dump artifacts | Illustrative producer-side parsing |
| Kernel, shape, architecture, and source fingerprint | Provisional producer-local key |
| Bundle schema and routing map | Provisional producer-side data structures |
| Applying a matched ACF and defining fallback behavior | Not implemented |
| Consumer compatibility and schema evolution | Not defined |

The example demonstrates producer-side fingerprint computation and bundle
lookup, including fail-closed fingerprint recomputation during evaluation. A
downstream XLA consumer is not included. Such a consumer would need to compute
the same key from the same canonical inputs, but this example does not validate
that integration or establish the fingerprint, bundle versioning, ACF
application, fallback, or compatibility behavior as stable cross-project
contracts.

## 1. Inspect an XLA dump

Run the source module in place:

```bash
export PYTHONPATH="$PWD/src"
python -m xla_ciq.cli xla inspect <dump-directory>
python -m xla_ciq.cli xla inspect <dump-directory> \
  --kernel-name <function-name> \
  --json
```

Inspection discovers PTX files, kernel names, shape tokens, launch metadata,
and tensor-map information represented by this example. It is not an XLA dump
or consumer compatibility contract.

## 2. Populate a bundle

Mock mode is the default. It exercises parsing, fingerprinting, producer-side
routing, and serialization without CompileIQ or a GPU:

```bash
python -m xla_ciq.cli bundle populate ./kernels.xla-ciq.bundle \
  --kernel-name fusion_demo \
  --input tests/fixtures/sample.ptx \
  --shapes 'f32[1024],f32[2048]' \
  --mock
```

For all shapes associated with a kernel in a dump:

```bash
python -m xla_ciq.cli bundle populate ./kernels.xla-ciq.bundle \
  --kernel-name fusion_demo \
  --input tests/fixtures/xla_dump/module.ptx \
  --shapes all \
  --dump-dir tests/fixtures/xla_dump \
  --mock
```

Real timing requires an explicit `--real` flag:

```bash
python -m xla_ciq.cli bundle populate ./kernels.xla-ciq.bundle \
  --kernel-name <kernel> \
  --input <kernel.ptx> \
  --shapes 'f32[1024]' \
  --dump-dir <dump-directory> \
  --generations 5 \
  --pool-size 32 \
  --task-timeout 30 \
  --num-workers 1 \
  --real
```

The real path creates independent, zero-filled buffers for pointer parameters.
It rejects scalar value parameters because it cannot reconstruct their values.
It also does not compare kernel outputs with expected results. The resulting
measurements and ACFs are illustrative and are not correctness-validated.

`xla_ciq.search.compileiq_runtime` adapts the example's `SearchRequest` and
`SearchResult` interface to the parent CompileIQ checkout. It selects
CompileIQ's isolated worker so each candidate runs in a killable process.
`--task-timeout` remains a per-objective timeout, not an overall search
deadline.

Each shape produces a separate provisional fingerprint and search. The
producer preserves an existing mapping by default, while `--replace` permits
replacement. A re-measured winner that is not faster than the baseline is
omitted unless `--keep-slower-acf` is set. An absent mapping says nothing about
how a future consumer would behave.

## 3. TMA descriptor layouts

TMA kernels may require descriptor information that PTX does not encode. Seed
layouts can be recovered from the dump or provided with `--tensor-maps`.

```bash
python -m xla_ciq.cli bundle populate ./kernels.xla-ciq.bundle \
  --kernel-name <tma-kernel> \
  --input <kernel.ptx> \
  --shapes 'bf16[8,2048]' \
  --dump-dir <dump-directory> \
  --tensor-maps <tensor-maps.json> \
  --sweep-tensor-map-tiles \
  --task-timeout 30 \
  --real
```

Tile sweeping can search legal box and swizzle settings with PTXAS controls.
The current bundle does not serialize a winning descriptor layout. The
producer therefore fails before saving if a swept winner differs from the seed
box or swizzle. A seed-equivalent winner may proceed.

## 4. Inspect the bundle

```bash
python -m xla_ciq.cli bundle inspect ./kernels.xla-ciq.bundle
python -m xla_ciq.cli bundle inspect ./kernels.xla-ciq.bundle --json
```

The bundle is a provisional
`xla_ciq.bundle.v1alpha1.XlaCiqTuningBundle`. Its current fields are:

1. `format_version`
2. `fingerprint_to_control_file_id`
3. `control_files`
4. `fingerprint_to_metadata`

The producer content-addresses ACF bytes by SHA-256 and garbage-collects
unreferenced entries after replacement or removal. Reading or writing fails on
an unsupported version, an incorrect content hash, a missing route target, or
metadata that has no routing mapping.

## 5. Re-measure selected entries

Mock evaluation checks producer-side selection without compiling or launching:

```bash
python -m xla_ciq.cli bundle evaluate ./kernels.xla-ciq.bundle \
  --input tests/fixtures/sample.ptx \
  --kernel-name fusion_demo \
  --mock \
  --json
```

Real re-measurement requires `--real`, a compatible CUDA environment, and
adequate launch metadata. It compares synthetic launch timing with and without
the selected ACF. It does not validate kernel outputs or define downstream
consumer behavior. Fingerprint mismatch is fatal rather than selecting an ACF
for different producer inputs.

## Failure and write behavior

- Invalid input, shape, bundle version, route, or content hash fails before search.
- A scalar parameter without an explicit value is rejected before a real launch.
- An unlaunchable shape is left unmapped; consumer fallback is unspecified.
- A search error stops the invocation.
- Bundle writes use a temporary file followed by replacement.
- A non-seed TMA winner is rejected before bundle mutation is saved.
- Real GPU, output-correctness, and downstream consumer behavior are outside CI.

## Example status

This source is an experimental reference example. It provides no supported XLA
integration, stable bundle or fingerprint contract, compatibility promise, or
roadmap commitment. A future integrator must define and validate those pieces.
