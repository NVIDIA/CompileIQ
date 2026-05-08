from unittest.mock import MagicMock

import pytest

from compileiq.ciq import Search
from compileiq.search_spaces import base as ss
from compileiq.types import SearchConfiguration


def _patch_search_internals(mocker):
    mocker.patch("compileiq.core.core_comms.socket.socket", return_value=MagicMock())
    mocker.patch("compileiq.worker.multiprocessing.Manager", return_value=MagicMock())


def _search_space():
    return {"lr": ss.range(start=0, end=10)}


class TestObjectiveConstructors:
    def test_single_objective_sets_objective_count(self, mocker, tmp_path):
        _patch_search_internals(mocker)

        search = Search.single_objective(
            objective_function=lambda p: 1.0,
            search_space=_search_space(),
            search_config=SearchConfiguration(generations=2),
            cache_folder=tmp_path,
        )

        assert search.objective_mode == "single"
        assert search._search_config.num_objectives == 1

    def test_multi_objective_sets_objective_count(self, mocker, tmp_path):
        _patch_search_internals(mocker)

        search = Search.multi_objective(
            objective_function=lambda p: (1.0, 2.0),
            search_space=_search_space(),
            search_config=SearchConfiguration(generations=2),
            num_objectives=2,
            cache_folder=tmp_path,
        )

        assert search.objective_mode == "multi"
        assert search._search_config.num_objectives == 2

    def test_multi_objective_recomputes_derived_cull_size(self, mocker, tmp_path):
        _patch_search_internals(mocker)

        search = Search.multi_objective(
            objective_function=lambda p: (1.0, 2.0),
            search_space=_search_space(),
            search_config=SearchConfiguration(
                pool_size=12,
                generations=3,
                mutate_rate=0.5,
                problem_type="min",
            ),
            num_objectives=2,
            cache_folder=tmp_path,
        )

        assert search._search_config.pool_size == 12
        assert search._search_config.cull_size == 6

    def test_multi_objective_recomputes_derived_pool_size(self, mocker, tmp_path):
        _patch_search_internals(mocker)
        expected_config = SearchConfiguration(generations=2, num_objectives=20)

        search = Search.multi_objective(
            objective_function=lambda p: tuple(float(i) for i in range(20)),
            search_space=_search_space(),
            search_config=SearchConfiguration(generations=2),
            num_objectives=20,
            cache_folder=tmp_path,
        )

        assert search._search_config.pool_size == expected_config.pool_size
        assert search._search_config.cull_size == expected_config.cull_size

    def test_multi_objective_rejects_single_objective_count(self, mocker, tmp_path):
        _patch_search_internals(mocker)

        with pytest.raises(ValueError, match="greater than 1"):
            Search.multi_objective(
                objective_function=lambda p: (1.0,),
                search_space=_search_space(),
                search_config=SearchConfiguration(generations=2),
                num_objectives=1,
                cache_folder=tmp_path,
            )

    def test_single_objective_rejects_explicit_mismatch(self, mocker, tmp_path):
        _patch_search_internals(mocker)

        with pytest.raises(ValueError, match="expected 1"):
            Search.single_objective(
                objective_function=lambda p: 1.0,
                search_space=_search_space(),
                search_config=SearchConfiguration(generations=2, num_objectives=2),
                cache_folder=tmp_path,
            )

    def test_multi_objective_rejects_dict_mismatch(self, mocker, tmp_path):
        _patch_search_internals(mocker)

        with pytest.raises(ValueError, match="expected 3"):
            Search.multi_objective(
                objective_function=lambda p: (1.0, 2.0, 3.0),
                search_space=_search_space(),
                search_config={"generations": 2, "num_objectives": 2},
                num_objectives=3,
                cache_folder=tmp_path,
            )


class TestSearchConfigNormalization:
    def test_constructor_accepts_dict_search_config(self, mocker, tmp_path):
        _patch_search_internals(mocker)

        search = Search(
            objective_function=lambda p: 1.0,
            search_space=_search_space(),
            search_config={"generations": 2},
            cache_folder=tmp_path,
        )

        assert search.objective_mode == "single"
        assert search._search_config.num_objectives == 1
