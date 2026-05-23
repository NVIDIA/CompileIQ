# Search-Space Retrieval Testing

This page describes checks for validating a Search Space Catalog Release before
relying on it in a run. The same checks apply to a public GitHub Release and to
a local mirror used in restricted environments.

## Local Catalog Checks

These checks do not require GitHub release publication or network access after
the release inputs have been collected.

1. Install the release-prep helper dependencies:

   ```bash
   make install-release
   ```

2. Seed a repo-local staging directory:

   ```bash
   make setup-search-space-release
   source dist/search-space-release/current.env
   ```

3. Add, remove, or replace approved `.bin` files in `${SS_INPUT_DIR}`.

4. Reconcile the staged `.bin` files into catalog metadata:

   ```bash
   make update-search-space-catalog
   ```

5. Build and validate the local staging bundle:

   ```bash
   make build-search-space-release
   make check-search-space-staging
   ```

The generated `${SS_OUTPUT_DIR}` should contain:

```text
manifest.json
*.bin
SHA256SUMS.txt
release-body.md
```

The validator confirms that `manifest.json`, `SHA256SUMS.txt`, the `.bin`
assets, and the release body agree.

## Clean Download Check

After creating a draft GitHub Release, validate the uploaded asset set from a
clean download:

```bash
tmpdir="$(mktemp -d)"
gh release download "${SS_RELEASE_TAG}" \
  --repo NVIDIA/CompileIQ \
  --dir "$tmpdir"

make check-search-space-assets \
  SS_OUTPUT_DIR="$tmpdir" \
  SS_RELEASE_TAG="${SS_RELEASE_TAG}"
```

Expected output:

```text
PASS: Validated Search Space release assets for search-spaces-YYYY.MM.DD in <tmpdir>.
```

## Live Retrieval Check

A live retrieval check validates the same public GitHub path used by online
clients. The target release must contain `manifest.json`, every `.bin` asset
referenced by that manifest, and `SHA256SUMS.txt`.

Run a clean-cache retrieval check against the published release:

```bash
rm -rf ~/.cache/compileiq
SS_VALIDATED_DOWNLOAD_DIR="$tmpdir" poetry run python - <<'PY'
import os
import pathlib

from compileiq.search_spaces.manifest import SearchSpaceManifestModel
from compileiq.search_spaces.resolver import resolve_with_metadata

tag = os.environ["SS_RELEASE_TAG"]
manifest_path = pathlib.Path(os.environ["SS_VALIDATED_DOWNLOAD_DIR"]) / "manifest.json"
manifest = SearchSpaceManifestModel.model_validate_json(manifest_path.read_text())
for entry in manifest.entries:
    resolved = resolve_with_metadata(
        entry.compiler,
        entry.compiler_version,
        entry.variant,
        tag=tag,
    )
    print(resolved.path)
PY
```

The paths printed should point at freshly downloaded cache files for every entry
in the release metadata.

## What Unit Tests Cover

The unit and integration tests cover local mirrors, mocked GitHub downloads,
`latest` filtering, prerelease skipping, cache hits, corrupt cache replacement,
digest and size failures, metadata exposure, tracker logging, and the local
release-prep scripts.
