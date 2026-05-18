import json

import pytest

from compileiq.core.verify_core import sha256_file, verify, verify_binary, verify_binary_platform


def _write_manifest(path, files):
    path.write_text(
        json.dumps(
            {
                "core_commit": "test-commit",
                "files": files,
            }
        ),
        encoding="utf-8",
    )


def test_verify_accepts_valid_manifest(tmp_path):
    root = tmp_path / "executable"
    binary = root / "linux" / "x86_64" / "bin" / "core"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"core")
    manifest = root / "core-manifest.json"
    _write_manifest(manifest, {"linux/x86_64/bin/core": f"sha256:{sha256_file(binary)}"})

    result = verify(root, manifest, required_platforms=())

    assert result.ok
    assert result.matches == ["linux/x86_64/bin/core"]


def test_verify_rejects_modified_binary(tmp_path):
    root = tmp_path / "executable"
    binary = root / "linux" / "x86_64" / "bin" / "core"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"core")
    manifest = root / "core-manifest.json"
    _write_manifest(manifest, {"linux/x86_64/bin/core": f"sha256:{sha256_file(binary)}"})

    binary.write_bytes(b"modified")
    result = verify(root, manifest, required_platforms=())

    assert not result.ok
    assert result.mismatches[0][0] == "linux/x86_64/bin/core"


def test_verify_binary_rejects_manifest_mismatch(tmp_path):
    root = tmp_path / "executable"
    binary = root / "linux" / "x86_64" / "bin" / "core"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"core")
    manifest = root / "core-manifest.json"
    _write_manifest(manifest, {"linux/x86_64/bin/core": f"sha256:{sha256_file(binary)}"})

    binary.write_bytes(b"modified")

    with pytest.raises(RuntimeError, match="does not match"):
        verify_binary(binary, root, manifest)


def test_verify_binary_platform_rejects_dependency_mismatch(tmp_path):
    root = tmp_path / "executable"
    launcher = root / "linux" / "x86_64" / "bin" / "core"
    dependency = root / "linux" / "x86_64" / "bin" / "_core"
    library = root / "linux" / "x86_64" / "lib" / "libciq.so"
    for path in (launcher, dependency, library):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(path.name.encode())
    manifest = root / "core-manifest.json"
    _write_manifest(
        manifest,
        {
            "linux/x86_64/bin/core": f"sha256:{sha256_file(launcher)}",
            "linux/x86_64/bin/_core": f"sha256:{sha256_file(dependency)}",
            "linux/x86_64/lib/libciq.so": f"sha256:{sha256_file(library)}",
        },
    )
    library.write_bytes(b"modified")

    with pytest.raises(RuntimeError, match="linux/x86_64/lib/libciq.so"):
        verify_binary_platform(launcher, root, manifest)


def test_verify_reports_missing_required_platform(tmp_path):
    root = tmp_path / "executable"
    binary = root / "linux" / "x86_64" / "bin" / "core"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"core")
    manifest = root / "core-manifest.json"
    _write_manifest(manifest, {"linux/x86_64/bin/core": f"sha256:{sha256_file(binary)}"})

    result = verify(root, manifest, required_platforms=("linux/x86_64", "win32/amd64"))

    assert not result.ok
    assert result.missing_platforms == ["win32/amd64"]
