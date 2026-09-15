from __future__ import annotations

from pathlib import Path

from xla_ciq.xla.launch_config import (
    fallback_launch_config,
    parse_thunk_launches,
    resolve_launch_config,
    thunk_sequence_for_ptx,
)


def test_thunk_sequence_for_ptx(tmp_path: Path):
    ptx = tmp_path / "module_0001.211.ptx"
    thunk = tmp_path / "module_0001.thunk_sequence.txt"
    ptx.write_text("// ptx\n")
    thunk.write_text("kernel_a grid: [16,1,1] threads: [256,1,1]\n")
    assert thunk_sequence_for_ptx(ptx) == thunk


def test_parse_thunk_launches_with_shared_mem(tmp_path: Path):
    path = tmp_path / "mod.thunk_sequence.txt"
    path.write_text(
        "055: kCustomKernel [prev=054 (1)  | next=056 (1)  ] "
        "input_add_reduce_fusion grid: [16, 1, 1] threads: [256, 1, 1]  "
        "shared_memory: 1024 bytes\n"
        "070: kCustomKernel other_kernel grid: [2,4,1] threads: [128,1,1]\n"
    )
    rows = parse_thunk_launches(path)
    assert len(rows) == 2
    name, cfg = rows[0]
    assert name == "input_add_reduce_fusion"
    assert cfg.grid == (16, 1, 1)
    assert cfg.block == (256, 1, 1)
    assert cfg.shared_memory_bytes == 1024
    assert cfg.block_size == 256


def test_resolve_launch_prefers_sibling_and_reqntid(tmp_path: Path):
    ptx = tmp_path / "module_x.211.ptx"
    sibling = tmp_path / "module_x.thunk_sequence.txt"
    other = tmp_path / "module_y.thunk_sequence.txt"
    ptx.write_text("//\n")
    sibling.write_text(
        "k grid: [8,1,1] threads: [128,2,1]\n"
        "k grid: [4,1,1] threads: [256,1,1]\n"
    )
    other.write_text("k grid: [99,1,1] threads: [64,1,1]\n")

    cfg = resolve_launch_config(
        kernel_name="k",
        ptx_path=ptx,
        dump_dir=tmp_path,
        reqntid=(256, 1, 1),
    )
    assert cfg is not None
    assert cfg.grid == (4, 1, 1)
    assert cfg.block == (256, 1, 1)


def test_fallback_launch_config():
    cfg = fallback_launch_config(numel=1000, block_size=256)
    assert cfg.grid == (4, 1, 1)
    assert cfg.block == (256, 1, 1)
    assert "heuristic" in cfg.source
