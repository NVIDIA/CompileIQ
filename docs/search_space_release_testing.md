# Search-Space Retrieval Testing

This page describes checks for validating a search-space catalog before relying
on it in a run. The same checks apply to a public GitHub release and to a local
mirror used in restricted environments.

## Local Catalog Checks

These checks do not require GitHub releases or network access.

1. Install the release-prep helper dependencies:

   ```bash
   make install-release
   ```

2. Stage candidate `.bin` files outside the repo checkout.
3. Build the release catalog:

   ```bash
   make build-search-space-manifest \
     SS_ARTIFACTS_DIR=/path/to/search-space-bins \
     SS_TAG=search-spaces-YYYY.MM.DD-rc1
   ```

4. Build release notes from the generated catalog:

   ```bash
   make build-search-space-release-notes
   ```

5. Validate the manifest schema and Python behavior:

   ```bash
   make check-search-space-manifest-schema
   poetry run pytest tests/unit/search_spaces tests/integration/test_search_space_resolution.py -vvv
   ```

6. Exercise the offline resolver path by placing `manifest.json` and the
   referenced `.bin` files in one directory and setting:

   ```bash
   export CIQ_SEARCH_SPACES_DIR=/path/to/mirrored/search-spaces
   ```

## Live GitHub Retrieval Check

A live retrieval check validates the same public GitHub path used by online
clients. The target release must contain `manifest.json` plus every `.bin`
asset referenced by that manifest.

First, validate the uploaded asset set from a clean download:

```bash
tmpdir="$(mktemp -d)"
gh release download search-spaces-YYYY.MM.DD \
  --repo NVIDIA/CompileIQ \
  --dir "$tmpdir"

cd "$tmpdir"
jq -r '.tag' manifest.json
jq -r '.entries[].filename' manifest.json | sort
find . -maxdepth 1 -name '*.bin' -exec basename {} \; | sort
```

The manifest filenames and downloaded `.bin` filenames must match exactly.

Recommended flow:

1. Choose the release repository. The default is `NVIDIA/CompileIQ`; set
   `CIQ_SEARCH_SPACES_REPO` only when validating a different release repo.
2. Choose the catalog tag to validate, such as `search-spaces-YYYY.MM.DD`.
3. Resolve with an explicit tag:

   ```python
   from compileiq.search_spaces.compilers import PtxasSearchSpace

   path = PtxasSearchSpace(version="13.3", tag="search-spaces-YYYY.MM.DD").retrieve()
   ```

4. Run the same retrieval a second time and confirm the verified cache is used.
5. For stable `search-spaces-*` releases, resolve with `tag="latest"` and
   confirm it selects the intended stable release. Drafts and prereleases are
   intentionally skipped by `latest`.
6. Confirm `Search.search_space_resolution_metadata` and tracker metadata record
   the resolved tag, filename, SHA256, size, source, and local path.

## What Unit Tests Cover

The unit and integration tests cover local mirrors, mocked GitHub downloads,
`latest` filtering, prerelease skipping, cache hits, corrupt cache replacement,
digest and size failures, metadata exposure, and tracker logging.
