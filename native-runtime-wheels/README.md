# Native runtime wheels

This directory is part of the public CompileIQ source tree. A Git checkout with
Git LFS contains the compiled native runtime needed by `make install-complete`
and by release-wheel composition. Provider source code is not included.

Install Git LFS and fetch the checked-in wheels before using them:

```bash
git lfs install
git lfs pull
make install-complete
```

`make install-complete` selects the single manifest and wheel compatible with
the active Python and platform, verifies the wheel's filename, size, and SHA-256
digest, and then installs it. Explicit artifact overrides remain available:

```bash
make install-complete \
  NATIVE_RUNTIME_WHEEL=/path/to/compileiq_native_runtime.whl \
  NATIVE_RUNTIME_MANIFEST=/path/to/native-runtime-manifest.json
```

## Included cells

| Platform | Python | Runtime wheel tag |
| --- | --- | --- |
| Linux x86_64 | CPython 3.11 | `cp311-cp311-linux_x86_64` |
| Linux x86_64 | CPython 3.12 | `cp312-cp312-linux_x86_64` |
| Linux x86_64 | CPython 3.13 | `cp313-cp313-linux_x86_64` |
| Linux x86_64 | CPython 3.14 | `cp314-cp314-linux_x86_64` |
| Linux aarch64 | CPython 3.11 | `cp311-cp311-linux_aarch64` |
| Linux aarch64 | CPython 3.12 | `cp312-cp312-linux_aarch64` |
| Linux aarch64 | CPython 3.13 | `cp313-cp313-linux_aarch64` |
| Linux aarch64 | CPython 3.14 | `cp314-cp314-linux_aarch64` |
| Windows AMD64 | CPython 3.11 | `cp311-cp311-win_amd64` |
| Windows AMD64 | CPython 3.12 | `cp312-cp312-win_amd64` |
| Windows AMD64 | CPython 3.13 | `cp313-cp313-win_amd64` |
| Windows AMD64 | CPython 3.14 | `cp314-cp314-win_amd64` |
| macOS 15+ arm64 | CPython 3.11 | `cp311-cp311-macosx_15_0_arm64` |
| macOS 15+ arm64 | CPython 3.12 | `cp312-cp312-macosx_15_0_arm64` |
| macOS 15+ arm64 | CPython 3.13 | `cp313-cp313-macosx_15_0_arm64` |
| macOS 15+ arm64 | CPython 3.14 | `cp314-cp314-macosx_15_0_arm64` |

Every wheel is an exact CI output promoted without renaming or rebuilding it.
The checked-in manifests retain the opaque producing pipeline and job IDs plus
the producer source commit, alongside public compatibility and digest data. They
omit private URLs, branch names, repository names, and upstream implementation
details. Full input provenance remains with the release pipeline and is verified
before promotion.

Users still install one complete `compileiq` distribution. The checked-in
runtime wheels are build inputs, not a second package users must manage. If the
platform matrix becomes unwieldy, the runtime can later be published as a
separate dependency without changing the public search API.
