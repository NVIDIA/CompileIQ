"""Tests for dev/build_manifest.py."""

from __future__ import annotations

import hashlib
import importlib.util
import pathlib
import sys

import pytest
from pydantic import ValidationError

# build_manifest.py lives in dev/ rather than under compileiq/ because it's a
# release-only tool, not part of the wheel. Import it via importlib so we
# don't depend on dev/ being on PYTHONPATH.
_DEV = pathlib.Path(__file__).resolve().parents[3] / "dev"
_SPEC = importlib.util.spec_from_file_location("build_manifest", _DEV / "build_manifest.py")
assert _SPEC is not None and _SPEC.loader is not None
build_manifest = importlib.util.module_from_spec(_SPEC)
sys.modules["build_manifest"] = build_manifest
_SPEC.loader.exec_module(build_manifest)


def _yaml(text: str, path: pathlib.Path) -> pathlib.Path:
    path.write_text(text)
    return path


def test_build_emits_correct_sha_and_size(tmp_path):
    bin_bytes = b"hello search space" * 1000
    (tmp_path / "ptxas13.3_search_space.bin").write_bytes(bin_bytes)
    src = _yaml(
        """
entries:
  - compiler: ptxas
    compiler_version: "13.3"
    variant: default
    filename: ptxas13.3_search_space.bin
""",
        tmp_path / "manifest-source.yaml",
    )

    manifest = build_manifest.build(src, tmp_path, "search-spaces-test")
    assert manifest.tag == "search-spaces-test"
    assert manifest.manifest_format == "1.0.0"
    assert len(manifest.entries) == 1
    entry = manifest.entries[0]
    assert entry.compiler == "ptxas"
    assert entry.compiler_version == "13.3"
    assert entry.variant == "default"
    assert entry.sha256 == hashlib.sha256(bin_bytes).hexdigest()
    assert entry.size_bytes == len(bin_bytes)


def test_build_threads_optional_fields(tmp_path):
    (tmp_path / "x.bin").write_bytes(b"x")
    src = _yaml(
        """
entries:
  - compiler: ptxas
    compiler_version: "13.3"
    variant: att
    filename: x.bin
    description: "PTX with attribute-based scheduling"
""",
        tmp_path / "manifest-source.yaml",
    )
    manifest = build_manifest.build(src, tmp_path, "tag")
    assert manifest.entries[0].description == "PTX with attribute-based scheduling"
    assert manifest.entries[0].variant == "att"


def test_build_defaults_missing_variant(tmp_path):
    (tmp_path / "x.bin").write_bytes(b"x")
    src = _yaml(
        """
entries:
  - compiler: ptxas
    compiler_version: "13.3"
    filename: x.bin
""",
        tmp_path / "manifest-source.yaml",
    )
    manifest = build_manifest.build(src, tmp_path, "tag")
    assert manifest.entries[0].variant == "default"


def test_build_rejects_duplicate_resolver_selector(tmp_path):
    (tmp_path / "x.bin").write_bytes(b"x")
    (tmp_path / "y.bin").write_bytes(b"y")
    src = _yaml(
        """
entries:
  - compiler: ptxas
    compiler_version: "13.3"
    filename: x.bin
  - compiler: ptxas
    compiler_version: "13.3"
    filename: y.bin
""",
        tmp_path / "manifest-source.yaml",
    )

    with pytest.raises(ValidationError, match="Duplicate manifest entry"):
        build_manifest.build(src, tmp_path, "tag")


def test_build_raises_if_referenced_file_missing(tmp_path):
    src = _yaml(
        """
entries:
  - compiler: ptxas
    compiler_version: "13.3"
    variant: default
    filename: missing.bin
""",
        tmp_path / "manifest-source.yaml",
    )
    with pytest.raises(FileNotFoundError, match="missing.bin"):
        build_manifest.build(src, tmp_path, "tag")


@pytest.mark.parametrize("filename", ["../secret.bin", "/tmp/secret.bin", "nested/secret.bin"])
def test_build_rejects_path_like_filename_before_reading(tmp_path, filename):
    outside = tmp_path.parent / "secret.bin"
    outside.write_bytes(b"outside staging")
    src = _yaml(
        f"""
entries:
  - compiler: ptxas
    compiler_version: "13.3"
    filename: {filename}
""",
        tmp_path / "manifest-source.yaml",
    )

    with pytest.raises(ValueError, match="filename"):
        build_manifest.build(src, tmp_path, "tag")


def test_build_rejects_invalid_yaml_shape(tmp_path):
    src = _yaml("just a string", tmp_path / "manifest-source.yaml")
    with pytest.raises(ValueError, match="entries"):
        build_manifest.build(src, tmp_path, "tag")


def test_build_rejects_unknown_entry_field(tmp_path):
    """Pydantic extra='forbid' on SearchSpaceEntry catches typos in the YAML."""
    (tmp_path / "x.bin").write_bytes(b"x")
    src = _yaml(
        """
entries:
  - compiler: ptxas
    compiler_version: "13.3"
    variant: default
    filename: x.bin
    typo_field: oops
""",
        tmp_path / "manifest-source.yaml",
    )
    with pytest.raises(ValidationError):
        build_manifest.build(src, tmp_path, "tag")


def test_repo_manifest_source_yaml_parses_and_builds(tmp_path):
    """The checked-in manifest-source.yaml is structurally valid and `build_manifest`
    produces a non-empty catalog when every declared filename exists in the artifacts dir.

    This does NOT validate real release bytes -- the artifacts here are throwaway
    placeholders. Real digest/size validation happens in the release workflow when
    the actual approved encrypted .bin files are staged.
    """
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    source = repo_root / "release" / "search-spaces" / "manifest-source.yaml"
    for filename in (
        "ptxas13.3_search_space.bin",
        "ptxas13.3_att_search_space.bin",
        "nvcc13.3_search_space.bin",
    ):
        (tmp_path / filename).write_bytes(f"fixture {filename}".encode())

    manifest = build_manifest.build(
        source,
        tmp_path,
        "search-spaces-smoke-test",
    )
    assert len(manifest.entries) >= 1
    for entry in manifest.entries:
        assert len(entry.sha256) == 64
        assert entry.size_bytes > 0
