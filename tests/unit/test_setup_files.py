"""
Tests for compileiq/utils/_setup_files.py.
"""

import pathlib

from compileiq.utils._setup_files import setup_search_space


class TestMultiConfigFilenames:
    def test_three_config_paths_do_not_compound(self, tmp_path):
        sources = []
        for i in range(3):
            src = tmp_path / f"source_{i}.config"
            src.write_text(f"; test config {i}\n")
            sources.append(src)

        target = tmp_path / "dna.config"
        result = setup_search_space(sources, str(target))

        assert isinstance(result, list)
        names = [pathlib.Path(path).name for path in result]
        assert names == ["0_dna.config", "1_dna.config", "2_dna.config"]

    def test_single_config_uses_base_filename(self, tmp_path):
        src = tmp_path / "source.config"
        src.write_text("; test\n")

        target = tmp_path / "dna.config"
        result = setup_search_space([src], str(target))

        assert result == str(target)
