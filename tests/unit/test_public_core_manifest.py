import importlib.util
from pathlib import Path

from compileiq.core.verify_core import compute_core_lock


def _render_public_manifest(source, source_manifest_sha256):
    script = Path(__file__).resolve().parents[2] / "dev" / "public_core_manifest.py"
    spec = importlib.util.spec_from_file_location("public_core_manifest", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.render_public_manifest(source, source_manifest_sha256)


def test_public_manifest_uses_public_fields():
    source = {
        "schema_version": 2,
        "core_commit": "abc123",
        "core_ref": "main",
        "core_tag": None,
        "built_at": "2026-06-03T21:08:53Z",
        "pipeline_id": 53580480,
        "files": {
            "linux/x86_64/bin/core": "sha256:" + "1" * 64,
        },
        "vault_artifacts": {"linux/x86_64": {"pipeline_id": 123}},
        "unexpected_internal_field": "do-not-publish",
    }

    public = _render_public_manifest(source, "2" * 64)

    assert "vault_artifacts" not in public
    assert "unexpected_internal_field" not in public
    assert public["source_manifest_sha256"] == "sha256:" + "2" * 64
    assert public["core_lock"] == compute_core_lock(public)
