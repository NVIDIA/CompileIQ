from __future__ import annotations

import json
from pathlib import Path

from xla_ciq.cli import main
from xla_ciq.xla.dump import inspect_dump

FIXTURES = Path(__file__).parent / "fixtures" / "xla_dump"


def test_inspect_dump_recovers_shapes():
    result = inspect_dump(FIXTURES, kernel_name="fusion_demo")
    assert len(result.kernels) == 1
    kernel = result.kernels[0]
    assert kernel.kernel_name == "fusion_demo"
    assert any(p.endswith("module.ptx") for p in kernel.ptx_files)
    assert "fusion.1" in kernel.profile_annotations
    assert "f32[1024,2048]" in kernel.shapes
    assert "f32[1024]" in kernel.shapes


def test_inspect_dump_skips_non_dtype_brackets(tmp_path: Path):
    (tmp_path / "module.ptx").write_text(".visible .entry demo(\n")
    (tmp_path / "module.txt").write_text(
        "demo = f32[8,4] fusion(args[0], data[1]), calls=demo\n"
    )
    kernel = inspect_dump(tmp_path, kernel_name="demo").kernels[0]
    assert "f32[8,4]" in kernel.shapes
    assert not [s for s in kernel.shapes if s.startswith(("args[", "data["))]


def test_cli_xla_inspect_json(capsys):
    code = main(["xla", "inspect", str(FIXTURES), "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["kernels"][0]["kernel_name"] == "fusion_demo"


def test_cli_xla_inspect_missing_kernel():
    code = main(
        ["xla", "inspect", str(FIXTURES), "--kernel-name", "does_not_exist"]
    )
    assert code == 2
