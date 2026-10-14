# Complete wheel release composition

CompileIQ 1.1 has one public installation contract:

```bash
pip install compileiq
```

Linux and Windows complete wheels contain both the bundled `ruby` backend and
the native `amber` backend. The Apple Silicon macOS 15+ beta wheels are
intentionally coreless because no macOS legacy core is shipped; they provide the
compiled `amber` backend. The native build may produce an independently auditable
`compileiq-native-runtime` wheel, but that wheel is an intermediate artifact,
not a second package for users to install.

## Release graph

The release lane performs these steps in order:

1. Build the frontend wheel containing the public API and, where available, the
   matching bundled core. Record its filename, size, SHA-256 digest, and source
   commit.
2. Build and audit the native runtime wheel, then bind it to the native source
   and static-library provenance in the lane manifest.
3. Run `dev/compose_compileiq_wheel.py`. It verifies both input wheels against
   the lane manifest before copying the native import package, backend entry
   point, and runtime-only dependencies into the frontend wheel.
4. Retag the output with the native platform tag, mark it as non-pure, and
   regenerate its wheel `RECORD`. A composition manifest binds the resulting
   wheel back to both verified inputs.
5. Run `dev/verify_complete_wheel.py`, followed by
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

## Editable development

`make install` installs the editable frontend with the bundled `ruby` engine
only. `make install-complete` additionally materializes the pinned native payload
from a wheel plus its lane manifest; the manifest is verified before
installation. Native-runtime developers can explicitly set `NATIVE_RUNTIME_SOURCE`
to build and audit a local payload instead.

The public source tree carries one generated manifest and CI-produced runtime
wheel per platform and Python cell through Git LFS. `make install-complete`
consumes a compatible checked-in artifact only after
verifying its filename, size, digest, and successful backend registration. If a
verified compatible artifact is unavailable, the complete install fails rather
than creating a ruby-only environment. Provider-source build instructions and
artifact transport coordinates remain with the provider and release systems;
they are not part of the public installation contract.
