"""Public-contract integration coverage for an installed compiled backend."""

from __future__ import annotations

import pytest

from compileiq.backends import get_backend_registration
from compileiq.ciq import Search
from compileiq.search_spaces import base as ss
from compileiq.types import SearchConfiguration
from compileiq.utils.validation import Score
from compileiq.worker import ManualWorker


def _scores(candidates):
    return [
        Score(
            score=float(index),
            params=candidate.params,
            param_id=candidate.param_id,
            num_objectives=1,
        )
        for index, candidate in enumerate(candidates)
    ]


@pytest.mark.requires_amber
def test_compiled_backend_ask_tell_and_checkpoint_continuation(tmp_path):
    """Exercise compiled registration and continuation only through public APIs."""

    registration = get_backend_registration("amber")
    assert registration.capabilities.checkpoint
    assert registration.capabilities.sealed_checkpoint

    search = Search(
        objective_function=lambda params: float(params["x"]),
        search_space={"x": ss.range(0, 100)},
        search_config=SearchConfiguration(generations=2, pool_size=6),
        worker_type=ManualWorker,
        engine="amber",
        cache_folder=tmp_path / "initial",
        disable_progress_bar=True,
    )
    first = search.ask()
    search.tell(_scores(first))
    checkpoint = search.save_checkpoint(
        tmp_path / "native.checkpoint",
        require_no_pending=True,
    )

    restored = Search.from_checkpoint(
        checkpoint,
        objective_function=lambda params: float(params["x"]),
        search_space={"x": ss.range(0, 100)},
        worker_type=ManualWorker,
        cache_folder=tmp_path / "restored",
        disable_progress_bar=True,
    )
    assert len(restored.result.get_results()) == 6
    second = restored.ask()
    assert {str(candidate.param_id) for candidate in first}.isdisjoint(
        str(candidate.param_id) for candidate in second
    )
    restored.tell(_scores(second))
    assert len(restored.result.get_results()) == 12
