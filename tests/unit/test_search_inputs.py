import sys
from unittest.mock import MagicMock

import pytest

from compileiq.ciq import Search, SearchSpaceInput
from compileiq.search_spaces import base as ss
from tests.utils import SCRIPTED_ENGINE

pytestmark = pytest.mark.usefixtures("scripted_engine")


def _patch_search_internals(mocker):
    mocker.patch("compileiq.worker.multiprocessing.Manager", return_value=MagicMock())


def _search_space():
    return {"lr": ss.range(start=0, end=10)}


class TestSearchConfigNormalization:
    def test_constructor_accepts_dict_search_config(self, mocker, tmp_path):
        _patch_search_internals(mocker)

        search = Search(
            objective_function=lambda p: 1.0,
            search_space=_search_space(),
            search_config={"generations": 2},
            cache_folder=tmp_path,
            engine=SCRIPTED_ENGINE,
        )

        assert search._search_config.num_objectives == 1


class TestSearchSpaceLists:
    def test_list_search_space_is_accepted_on_windows(self, mocker, monkeypatch, tmp_path):
        _patch_search_internals(mocker)
        monkeypatch.setattr(sys, "platform", "win32")
        search_space: SearchSpaceInput = [_search_space(), {"tile": ss.range(start=1, end=8)}]

        search = Search(
            objective_function=lambda params: 1.0,
            search_space=search_space,
            search_config={"generations": 1},
            cache_folder=tmp_path,
            engine=SCRIPTED_ENGINE,
        )

        assert search.search_space == search_space
