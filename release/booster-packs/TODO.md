# Booster Pack Release Follow-Up

This is a follow-up marker for Booster Pack release infrastructure. The current
search-space retrieval work intentionally supports release-backed `.bin` search
spaces only. Booster Packs are related because they should use the same release
discipline, but they are not inputs to the search-space resolver and should not
be folded into the search-space manifest contract.

If Booster Packs share CompileIQ release infrastructure, add first-class
manifest/data-model support here rather than letting ad hoc zip layouts become
the contract.

## TODO

- Define the Booster Pack catalog schema used at the GitHub release level.
- Define the per-pack manifest schema stored inside each Booster Pack zip.
- Add release-prep scripts that compute `sha256` and size metadata for pack
  zips and candidate `.acf` files.
- Add Markdown release-note generation for Booster Pack catalog releases.
- Add schema parity tests, malformed-manifest tests, and duplicate-candidate
  tests before publishing real Booster Pack assets.
- Decide later whether package runtime download helpers are needed. Do not add
  a runtime API until the manual release and documentation workflow has settled.

## Constraints

- Keep Booster Packs separate from search-space catalogs.
- Reserve top-level release `manifest.json` for search-space `.bin` catalogs.
- Put the Booster Pack manifest inside each zip, or name any release-level file
  `booster-pack-manifest.json`.
- Do not use the `search-spaces-*` tag namespace for Booster Pack releases.
- Treat Booster Packs as already-generated `.acf` candidate bundles, not
  inputs to `PtxasSearchSpace` or `NvccSearchSpace`.

## Proposed Release Shape

A Booster Pack catalog release should use a Booster Pack-specific tag prefix,
for example `booster-packs-2026.05.12`. Top-level release assets should be
directly fetchable through GitHub release APIs.

```text
booster-packs-2026.05.12/
  booster-pack-catalog.json
  helion-booster-pack.zip
  triton-gemm-booster-pack.zip
  flashinfer-booster-pack.zip
  debug-pack.zip
```

The top-level `booster-pack-catalog.json` lists the supported packs in that
catalog version and records each zip filename, size, and digest. Each zip then
contains a pack-specific manifest plus the candidate ACFs.

```text
helion-booster-pack.zip
  booster-pack-manifest.json
  candidates/
    candidate-001.acf
    candidate-002.acf
```

## Data Model Sketch

The pieces that matter are the stable selectors, per-candidate integrity
metadata, and validation context. A future runtime or release tool can use a
model shaped like this:

```python
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class BoosterPackCandidate(BaseModel, extra="forbid"):
    filename: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(gt=0)
    description: str | None = None
    source_workload: str | None = None
    expected_effect: Literal["improvement", "diagnostic", "mixed", "unknown"] = "unknown"


class BoosterPackValidationTarget(BaseModel, extra="forbid"):
    workload: str
    framework: str | None = None
    compiler: Literal["ptxas", "nvcc", "mixed"]
    compiler_version: str
    gpu_target: str | None = None
    cuda_toolkit: str | None = None
    command: str | None = None
    notes: str | None = None


class BoosterPackManifestModel(BaseModel, extra="forbid"):
    manifest_format: Literal["booster-pack-manifest-v1"] = "booster-pack-manifest-v1"
    pack_slug: str
    pack_name: str
    release_tag: str
    generated_at: str
    controls_interface_min_ctk: str | None = None
    candidates: list[BoosterPackCandidate]
    validation_targets: list[BoosterPackValidationTarget] = []
    caveats: list[str] = []

    @model_validator(mode="after")
    def reject_duplicate_candidate_filenames(self) -> "BoosterPackManifestModel":
        seen: set[str] = set()
        for candidate in self.candidates:
            if candidate.filename in seen:
                raise ValueError(f"Duplicate candidate filename: {candidate.filename}")
            seen.add(candidate.filename)
        return self
```

That keeps the existing search-space resolver contract intact while giving
Booster Packs a schema that can later support validation, mirroring, and
integrity checks.
