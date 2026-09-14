from __future__ import annotations

from pathlib import Path

import pytest

from xla_ciq.cli import main
from xla_ciq.bundle import load_bundle
from xla_ciq.fingerprint import content_id, fingerprint

FIXTURES = Path(__file__).parent / "fixtures"


def test_populate_creates_bundle(tmp_path: Path, capsys):
    bundle = tmp_path / "out.bundle"
    ptx = FIXTURES / "sample.ptx"
    code = main(
        [
            "bundle",
            "populate",
            str(bundle),
            "--kernel-name",
            "fusion_demo",
            "--input",
            str(ptx),
            "--shapes",
            "f32[1024],f32[2048]",
            "--mock",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "MOCK:" in captured.err
    assert bundle.is_file()
    loaded = load_bundle(bundle)
    assert len(loaded.fingerprint_to_control_file_id) == 2
    assert len(loaded.control_files) == 2
    assert captured.out.count("added:") == 2
    content_ids = set(loaded.control_files.keys())
    assert len(content_ids) == 2
    for fp, meta in loaded.fingerprint_to_metadata.items():
        assert meta.kernel_name == "fusion_demo"
        assert meta.shape in {"f32[1024]", "f32[2048]"}
        assert not meta.has_timing


def test_populate_preserves_without_replace(tmp_path: Path, capsys):
    bundle = tmp_path / "out.bundle"
    ptx = FIXTURES / "sample.ptx"
    args = [
        "bundle",
        "populate",
        str(bundle),
        "--kernel-name",
        "fusion_demo",
        "--input",
        str(ptx),
        "--shapes",
        "f32[1]",
        "--mock",
    ]
    assert main(args) == 0
    first = load_bundle(bundle)
    first_cid = next(iter(first.control_files))
    capsys.readouterr()  # clear first-run output

    # Same shape → same fingerprint → skip search and preserve mapping.
    assert main(args) == 0
    captured = capsys.readouterr()
    assert "preserved:" in captured.out
    assert "ACF already exists" in captured.err
    assert "search skipped" in captured.err
    assert "--replace" in captured.err
    assert "MOCK:" not in captured.err
    second = load_bundle(bundle)
    assert next(iter(second.control_files)) == first_cid
    assert len(second.fingerprint_to_control_file_id) == 1


def test_populate_skips_existing_runs_new_shapes(tmp_path: Path, capsys):
    bundle = tmp_path / "out.bundle"
    ptx = FIXTURES / "sample.ptx"
    assert (
        main(
            [
                "bundle",
                "populate",
                str(bundle),
                "--kernel-name",
                "fusion_demo",
                "--input",
                str(ptx),
                "--shapes",
                "f32[1]",
                "--mock",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert (
        main(
            [
                "bundle",
                "populate",
                str(bundle),
                "--kernel-name",
                "fusion_demo",
                "--input",
                str(ptx),
                "--shapes",
                "f32[1],f32[2]",
                "--mock",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert "ACF already exists for shape=f32[1]" in captured.err
    assert "search skipped" in captured.err
    assert "MOCK: shape=f32[2]" in captured.err
    assert "MOCK: shape=f32[1]" not in captured.err
    assert "preserved:" in captured.out
    assert "added:" in captured.out
    loaded = load_bundle(bundle)
    assert len(loaded.fingerprint_to_control_file_id) == 2


def test_populate_different_shapes_are_distinct(tmp_path: Path):
    bundle = tmp_path / "out.bundle"
    ptx = FIXTURES / "sample.ptx"
    assert (
        main(
            [
                "bundle",
                "populate",
                str(bundle),
                "--kernel-name",
                "fusion_demo",
                "--input",
                str(ptx),
                "--shapes",
                "f32[1]",
                "--mock",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "bundle",
                "populate",
                str(bundle),
                "--kernel-name",
                "fusion_demo",
                "--input",
                str(ptx),
                "--shapes",
                "f32[2]",
                "--mock",
            ]
        )
        == 0
    )
    loaded = load_bundle(bundle)
    assert len(loaded.fingerprint_to_control_file_id) == 2
    assert len(loaded.control_files) == 2


def test_populate_replace(tmp_path: Path, capsys):
    bundle = tmp_path / "out.bundle"
    ptx = FIXTURES / "sample.ptx"

    def args(*extra: str) -> list[str]:
        return [
            "bundle",
            "populate",
            str(bundle),
            "--kernel-name",
            "fusion_demo",
            "--input",
            str(ptx),
            "--shapes",
            "f32[1]",
            "--mock",
            *extra,
        ]

    assert main(args()) == 0
    first_cid = next(iter(load_bundle(bundle).control_files))
    assert main(args("--replace")) == 0
    assert "replaced:" in capsys.readouterr().out
    loaded = load_bundle(bundle)
    assert len(loaded.fingerprint_to_control_file_id) == 1
    assert next(iter(loaded.control_files)) == first_cid


def test_populate_rejects_bad_input(tmp_path: Path):
    bundle = tmp_path / "out.bundle"
    bad = tmp_path / "x.txt"
    bad.write_text("nope")
    code = main(
        [
            "bundle",
            "populate",
            str(bundle),
            "--kernel-name",
            "k",
            "--input",
            str(bad),
            "--shapes",
            "f32[1]",
            "--mock",
        ]
    )
    assert code == 2
    assert not bundle.exists()


@pytest.mark.parametrize(
    ("removed_option", "value"),
    [
        ("--golden", "reference.bin"),
        ("--tolerance", "1e-3"),
        ("--timeout", "60"),
    ],
)
def test_populate_rejects_removed_options(
    tmp_path: Path,
    capsys,
    removed_option: str,
    value: str,
):
    bundle = tmp_path / "out.bundle"
    ptx = FIXTURES / "sample.ptx"
    code = main(
        [
            "bundle",
            "populate",
            str(bundle),
            "--kernel-name",
            "fusion_demo",
            "--input",
            str(ptx),
            "--shapes",
            "f32[1]",
            "--mock",
            removed_option,
            value,
        ]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert f"No such option '{removed_option}'" in captured.err
    assert not bundle.exists()


def test_populate_shapes_all_from_dump(tmp_path: Path, capsys):
    bundle = tmp_path / "out.bundle"
    dump = FIXTURES / "xla_dump"
    ptx = dump / "module.ptx"
    code = main(
        [
            "bundle",
            "populate",
            str(bundle),
            "--kernel-name",
            "fusion_demo",
            "--input",
            str(ptx),
            "--shapes",
            "all",
            "--dump-dir",
            str(dump),
            "--mock",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "resolved --shapes all" in captured.err
    assert "f32[1024,2048]" in captured.err
    assert "f32[1024]" in captured.err
    loaded = load_bundle(bundle)
    assert len(loaded.fingerprint_to_control_file_id) == 2
    assert captured.out.count("added:") == 2


def test_populate_shapes_all_requires_dump_dir(tmp_path: Path, capsys):
    bundle = tmp_path / "out.bundle"
    ptx = FIXTURES / "sample.ptx"
    code = main(
        [
            "bundle",
            "populate",
            str(bundle),
            "--kernel-name",
            "fusion_demo",
            "--input",
            str(ptx),
            "--shapes",
            "all",
            "--mock",
        ]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert "--dump-dir is required" in captured.err
    assert not bundle.exists()


def test_fingerprint_includes_shape():
    data = b"abc"
    a = fingerprint(
        cuda_version="13.3",
        arch="sm_90a",
        source_bytes=data,
        kernel_name="k",
        shape="f32[1]",
    )
    b = fingerprint(
        cuda_version="13.3",
        arch="sm_90a",
        source_bytes=data,
        kernel_name="k",
        shape="f32[1]",
    )
    c = fingerprint(
        cuda_version="13.3",
        arch="sm_90a",
        source_bytes=data,
        kernel_name="k",
        shape="f32[2]",
    )
    assert a == b
    assert a != c
    assert len(a) == 64
    assert content_id(b"x") != content_id(b"y")


def test_beats_baseline_policy():
    from xla_ciq.commands.populate import _beats_baseline

    assert _beats_baseline(1.0, 0.9) is True
    assert _beats_baseline(1.0, 1.0) is False
    assert _beats_baseline(1.0, 1.1) is False
    assert _beats_baseline(None, 0.5) is True


def test_populate_defaults_to_mock(tmp_path: Path, capsys):
    bundle = tmp_path / "default-mock.bundle"

    code = main(
        [
            "bundle",
            "populate",
            str(bundle),
            "--kernel-name",
            "fusion_demo",
            "--input",
            str(FIXTURES / "sample.ptx"),
            "--shapes",
            "f32[1024]",
        ]
    )

    assert code == 0
    assert bundle.is_file()
    assert "MOCK:" in capsys.readouterr().err


def test_populate_skips_acf_when_slower_than_baseline(tmp_path: Path, monkeypatch, capsys):
    from xla_ciq.search.types import SearchResult

    class SlowEngine:
        def run(self, request):
            return SearchResult(
                acf_bytes=b"slower-acf",
                notes=["REAL: synthetic slower"],
                score=0.20,
                baseline_ms=0.10,
            )

    monkeypatch.setattr(
        "xla_ciq.commands.populate.get_search_engine",
        lambda mock=False: SlowEngine(),
    )
    bundle = tmp_path / "out.bundle"
    ptx = FIXTURES / "sample.ptx"
    code = main(
        [
            "bundle",
            "populate",
            str(bundle),
            "--kernel-name",
            "fusion_demo",
            "--input",
            str(ptx),
            "--shapes",
            "f32[1024]",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "no_gain:" in captured.err
    assert "skipped_no_gain:" in captured.out
    loaded = load_bundle(bundle)
    assert len(loaded.fingerprint_to_control_file_id) == 0
    assert len(loaded.control_files) == 0


def test_populate_keep_slower_acf(tmp_path: Path, monkeypatch, capsys):
    from xla_ciq.search.types import SearchResult

    class SlowEngine:
        def run(self, request):
            return SearchResult(
                acf_bytes=b"slower-acf",
                notes=[],
                score=0.20,
                baseline_ms=0.10,
            )

    monkeypatch.setattr(
        "xla_ciq.commands.populate.get_search_engine",
        lambda mock=False: SlowEngine(),
    )
    bundle = tmp_path / "out.bundle"
    ptx = FIXTURES / "sample.ptx"
    code = main(
        [
            "bundle",
            "populate",
            str(bundle),
            "--kernel-name",
            "fusion_demo",
            "--input",
            str(ptx),
            "--shapes",
            "f32[1024]",
            "--keep-slower-acf",
        ]
    )
    assert code == 0
    loaded = load_bundle(bundle)
    assert len(loaded.fingerprint_to_control_file_id) == 1
    assert content_id(b"slower-acf") in loaded.control_files


def test_non_seed_tma_winner_leaves_existing_bundle_unchanged(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    from xla_ciq.search.compileiq_runtime import _require_seed_equivalent_winner
    from xla_ciq.xla.tensor_map import TensorMapSpec

    bundle = tmp_path / "out.xla-ciq.bundle"
    ptx = FIXTURES / "sample.ptx"
    base_args = [
        "bundle",
        "populate",
        str(bundle),
        "--kernel-name",
        "fusion_demo",
        "--input",
        str(ptx),
        "--shapes",
        "f32[1024]",
    ]
    assert main([*base_args, "--mock"]) == 0
    before = bundle.read_bytes()
    capsys.readouterr()

    seed = TensorMapSpec(dtype="f32", dims=(64,), box=(16,), swizzle_bytes=0)
    winner = TensorMapSpec(dtype="f32", dims=(64,), box=(32,), swizzle_bytes=0)

    class NonSeedWinnerEngine:
        def run(self, request):
            _require_seed_equivalent_winner(
                seed_maps={0: seed},
                winning_maps={0: winner},
                sweep_indices=(0,),
            )
            raise AssertionError("non-seed winner unexpectedly accepted")

    monkeypatch.setattr(
        "xla_ciq.commands.populate.get_search_engine",
        lambda mock=False: NonSeedWinnerEngine(),
    )
    code = main([*base_args, "--replace", "--sweep-tensor-map-tiles"])
    captured = capsys.readouterr()

    assert code == 1
    assert "winner=box=(32,),swizzle=0B" in captured.err
    assert "Bundle not modified" in captured.err
    assert bundle.read_bytes() == before


def test_seed_equivalent_tma_winner_is_serializable(tmp_path: Path, monkeypatch):
    from xla_ciq.search.compileiq_runtime import _require_seed_equivalent_winner
    from xla_ciq.search.types import SearchResult
    from xla_ciq.xla.tensor_map import TensorMapSpec

    seed = TensorMapSpec(dtype="f32", dims=(64,), box=(16,), swizzle_bytes=0)

    class SeedEquivalentWinnerEngine:
        def run(self, request):
            _require_seed_equivalent_winner(
                seed_maps={0: seed},
                winning_maps={0: seed},
                sweep_indices=(0,),
            )
            return SearchResult(acf_bytes=b"seed-equivalent-acf", notes=[])

    monkeypatch.setattr(
        "xla_ciq.commands.populate.get_search_engine",
        lambda mock=False: SeedEquivalentWinnerEngine(),
    )
    bundle = tmp_path / "out.xla-ciq.bundle"
    code = main(
        [
            "bundle",
            "populate",
            str(bundle),
            "--kernel-name",
            "fusion_demo",
            "--input",
            str(FIXTURES / "sample.ptx"),
            "--shapes",
            "f32[1024]",
            "--sweep-tensor-map-tiles",
        ]
    )

    assert code == 0
    loaded = load_bundle(bundle)
    assert content_id(b"seed-equivalent-acf") in loaded.control_files
