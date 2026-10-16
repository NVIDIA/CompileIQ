# Complete wheel release composition

CompileIQ 1.1 has one public installation contract:

```bash
pip install compileiq
```

Every complete wheel contains the pure-Python frontend and the compiled native
`amber` backend, which is the default and only bundled engine. No platform ships
separate core binaries, so Linux, Windows, and macOS wheels are composed the same way. The native build
may produce an independently auditable `compileiq-native-runtime` wheel, but that
wheel is an intermediate artifact, not a second package for users to install.

## Release graph

The release lane performs these steps in order:

1. Build the pure-Python frontend wheel containing the public API. Record its
   filename, size, SHA-256 digest, and source commit.
2. Build and audit the native runtime wheel, then bind it to the native source
   and static-library provenance in the lane manifest.
3. Run `dev/compose_compileiq_wheel.py`. It verifies both input wheels against
   the lane manifest before copying the native import package, backend entry
   point, and runtime-only dependencies into the frontend wheel.
4. Retag the output with the native platform tag, mark it as non-pure, and
   regenerate its wheel `RECORD`. A composition manifest binds the resulting
   wheel back to both verified inputs.
5. Run `dev/verify_complete_wheel.py`, which also rejects any file left over
   from the removed bundled core, followed by
   `dev/test_complete_wheel_install.py` in a clean environment. The acceptance
   environment installs only the composed `compileiq` wheel and its ordinary
   public dependencies.
6. Publish only wheels that passed that complete-wheel acceptance. Frontend-only
   wheels remain CI inputs and are never release assets.

The release matrix is fail-closed per platform. If a compatible native payload
cannot be built, audited, and composed for one platform and Python ABI, no
same-version wheel is published for that cell. Package installation therefore
fails normally instead of silently publishing an incomplete product for that
cell.

## Publishing from GitHub CI

Every composition input is in this repository, so GitHub CI
(`.github/workflows/ci.yml`) runs the release graph above for the 16 published
cells: CPython 3.11, 3.12, 3.13, and 3.14 on Linux x86_64, Linux aarch64,
Windows AMD64, and macOS 15+ arm64.

1. `build-and-smoke-test-wheel` builds the one pure-Python frontend wheel. On a
   `v*` tag its version is set from the tag.
2. `compose-complete-wheels` composes all 16 wheels from that frontend wheel
   and the native runtime wheels and manifests under `native-runtime-wheels/`.
   The checked-in manifests bind only the runtime wheel, so composition passes
   `--frontend-from-this-build`; the composition manifest still records the
   frontend wheel's digest.
3. `verify-and-test-complete-wheel` runs `dev/verify_complete_wheel.py` and
   `dev/test_complete_wheel_install.py` for each wheel on its own platform and
   Python version. On a tag, acceptance also checks the installed version
   against the tag. Pull requests run the Linux x86_64 cells; `main` and `v*`
   tags run all 16.
4. On a `v*` tag, `upload-github-release-wheels` uploads the 16 accepted
   wheels, and nothing else, to a draft GitHub Release for that tag. It fails
   unless exactly 16 accepted platform wheels are present. Publishing the draft
   and uploading to PyPI remain separate release steps.

## Editable development

`make install` installs the editable frontend and materializes the pinned native
payload from a wheel plus its lane manifest; the manifest is verified before
installation. `make install-complete` is an alias for it, and
`make install-frontend` installs the frontend alone, without an engine.
Native-runtime developers can explicitly set `NATIVE_RUNTIME_SOURCE` to build and
audit a local payload instead.

The public source tree carries one generated manifest and CI-produced runtime
wheel per platform and Python cell through Git LFS. `make install-complete`
consumes a compatible checked-in artifact only after
verifying its filename, size, digest, and successful backend registration. If a
verified compatible artifact is unavailable, the install fails rather than
creating an environment without an engine. Provider-source build instructions and
artifact transport coordinates remain with the provider and release systems;
they are not part of the public installation contract.
