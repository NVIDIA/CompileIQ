"""Adapter from the bundled core IPC protocol to the public backend contract."""

from __future__ import annotations

import json
import pathlib
import socket
from typing import cast

from compileiq.backends import BackendCandidate, SearchBackend
from compileiq.core.core_comms import CoreIPC, initialize_socket
from compileiq.core.core_types import (
    CompletionMessage,
    EvaluatedParamResponse,
    ParameterSet,
    ResponseTemplate,
)
from compileiq.types import InternalSearchConfiguration, ParamArg
from compileiq.utils._setup_files import (
    get_core_filepaths,
    setup_legacy_search_config,
    setup_search_space,
)
from compileiq.utils.validation import Score
from compileiq.utils.helpers import _decode_from_core, restore_nested_search_space


def _objective_values(value, num_objectives: int) -> list[int | float | str]:
    """Translate public scalar/tuple score shapes into the bundled IPC list shape."""

    if num_objectives > 1:
        if not isinstance(value, (list, tuple)):
            raise ValueError("A multi-objective score must be a list or tuple.")
        return list(value)
    if isinstance(value, (list, tuple)):
        raise ValueError("A single-objective score must be scalar.")
    return [value]


class BundledCoreBackend(SearchBackend):
    """Expose the shipped core through the same ask/tell lifecycle as plugins."""

    def __init__(
        self,
        *,
        search_space,
        search_config: InternalSearchConfiguration,
        cache_folder: pathlib.Path,
        debug: bool = False,
    ) -> None:
        """Prepare adapter state while deferring process and socket creation until ask()."""

        self._search_space = search_space
        self._search_config = search_config
        self._cache_folder = cache_folder
        self._multi_config = isinstance(search_space, list)
        self._file_backed_search_space: bool | list[bool]
        if self._multi_config:
            self._file_backed_search_space = [
                isinstance(config, pathlib.Path) for config in search_space
            ]
        else:
            self._file_backed_search_space = isinstance(search_space, pathlib.Path)
        self._debug = debug
        self._listen_socket: socket.socket | None = None
        self._core_socket: socket.socket | None = None
        self._core_ipc = CoreIPC()
        self._generation = 0
        self._complete = False
        self._started = False
        self._pending_ids: tuple[str, ...] = ()
        self._pending_candidates: dict[str, BackendCandidate] = {}
        self._core_ids_by_pending_id: dict[str, int] = {}
        self._buffered_scores: dict[str, Score] = {}
        self._batch_sequence = 0
        self.invocation_id: int | None = None

    @staticmethod
    def _parse_param_payload(param: str, *, is_file_backed: bool) -> ParamArg:
        """Parse one core payload while preserving opaque file-backed values."""

        try:
            param_set = json.loads(param)
        except (ValueError, json.JSONDecodeError):
            try:
                import json5

                param_set = json5.loads(param)
            except Exception:
                if not is_file_backed:
                    raise RuntimeError(
                        "Core returned an unparseable parameter payload for a "
                        "dictionary-defined CompileIQ search space. Expected a "
                        f"JSON object string, got: {param!r}"
                    ) from None
                return param

        if is_file_backed:
            return cast(ParamArg, param_set)
        if not isinstance(param_set, dict):
            raise RuntimeError(
                "Core returned a non-object parameter payload for a dictionary-defined "
                "CompileIQ search space. Expected a JSON object string, got "
                f"{type(param_set).__name__}: {param!r}"
            )
        return restore_nested_search_space(param_set)

    def _load_params(self, parameter_sets: ParameterSet) -> list[ParamArg]:
        """Translate bundled-core parameter payloads into public candidate values.

        This parser belongs to the bundled adapter because its JSON/JSON5 and
        base64 shapes are details of the core IPC protocol, not frontend search
        orchestration.
        """

        params_from_generation: list[ParamArg] = []
        for param in parameter_sets.params:
            if self._multi_config:
                assert isinstance(self._file_backed_search_space, list)
                decoded = [_decode_from_core(value) for value in json.loads(param.knobs)]
                value: ParamArg = cast(
                    ParamArg,
                    [
                        self._parse_param_payload(
                            single_param,
                            is_file_backed=self._file_backed_search_space[index],
                        )
                        for index, single_param in enumerate(decoded)
                    ],
                )
            else:
                assert isinstance(self._file_backed_search_space, bool)
                value = self._parse_param_payload(
                    param.knobs,
                    is_file_backed=self._file_backed_search_space,
                )
            params_from_generation.append(value)
        return params_from_generation

    @property
    def current_generation(self) -> int:
        """Return the generation most recently supplied by the bundled core."""

        return self._generation

    @property
    def pending_candidates(self) -> tuple[BackendCandidate, ...]:
        """Return unresolved bundled candidates without exposing adapter state."""

        return tuple(
            candidate.model_copy(deep=True)
            for candidate_id, candidate in self._pending_candidates.items()
            if candidate_id not in self._buffered_scores
        )

    @property
    def has_accepted_results_in_current_generation(self) -> bool:
        """Return whether an earlier tell subset contains a successful evaluation."""

        return any(not score.failed for score in self._buffered_scores.values())

    @property
    def is_complete(self) -> bool:
        """Return whether the bundled core emitted its terminal completion message."""

        return self._complete

    def _start(self) -> None:
        """Materialize legacy config and establish IPC exactly once on first use."""

        if self._started:
            return
        self._cache_folder.mkdir(parents=True, exist_ok=True)
        main_config, search_space_config = get_core_filepaths(self._cache_folder)
        self._search_config.dna_config = setup_search_space(
            self._search_space,
            search_space_config,
        )
        setup_legacy_search_config(self._search_config, main_config)
        self._listen_socket = initialize_socket()
        self._core_ipc.start(
            server_socket=self._listen_socket,
            main_config_filepath=main_config,
            silent=not self._debug,
        )
        self._core_socket, _ = self._listen_socket.accept()
        self._started = True

    def ask(self, num_samples: int | None = None) -> list[BackendCandidate]:
        """Receive one bundled-core generation and retain its candidate associations."""

        if num_samples not in (None, self._search_config.pool_size):
            raise ValueError("The bundled core does not support arbitrary ask batch sizes.")
        if self._pending_ids:
            raise RuntimeError("The bundled core still has pending candidates.")
        self._start()
        assert self._core_socket is not None
        message = self._core_ipc.receive_from_core(self._core_socket)
        if isinstance(message, CompletionMessage):
            if not message.complete:
                raise RuntimeError("The bundled core reported unsuccessful completion.")
            self._complete = True
            self._generation = self._search_config.generations
            return []

        params = self._load_params(message)
        if len(params) != len(message.params):
            raise RuntimeError("The bundled core returned inconsistent candidate payloads.")
        self._generation = message.generation_num
        self.invocation_id = message.invocation_id
        core_ids = [candidate.id for candidate in message.params]
        if len(core_ids) != len(set(core_ids)):
            raise RuntimeError("The bundled core returned duplicate candidate identifiers.")
        self._batch_sequence += 1
        self._pending_ids = tuple(
            f"bundled-{self._batch_sequence}-{index}-{candidate_id}"
            for index, candidate_id in enumerate(core_ids)
        )
        self._core_ids_by_pending_id = dict(zip(self._pending_ids, core_ids, strict=True))
        candidates = [
            BackendCandidate(param_id=candidate_id, params=params[index])
            for index, candidate_id in enumerate(self._pending_ids)
        ]
        self._pending_candidates = {str(candidate.param_id): candidate for candidate in candidates}
        return [candidate.model_copy(deep=True) for candidate in candidates]

    def tell(self, scores: list[Score]) -> None:
        """Buffer score subsets and send a complete core response in issued order.

        The public tell contract is ID-based and permits non-empty subsets in
        any order. The bundled IPC protocol still consumes one complete batch,
        so this adapter retains accepted subsets until every issued ID has a
        score. Every subset is validated before it changes adapter state. If
        response construction or the final IPC send fails, the final subset is
        still retryable and previously accepted subsets remain safely buffered.
        """

        if not scores:
            raise ValueError("Bundled-core tell requires at least one score.")
        score_ids = [str(score.param_id) for score in scores]
        duplicate_ids = sorted(
            {candidate_id for candidate_id in score_ids if score_ids.count(candidate_id) > 1}
        )
        if duplicate_ids:
            raise ValueError(
                f"Bundled-core tell contains duplicate candidate IDs: {duplicate_ids}."
            )
        pending_ids = set(self._pending_ids)
        unknown_ids = sorted(set(score_ids) - pending_ids)
        if unknown_ids:
            raise ValueError(f"Bundled-core tell contains unknown candidate IDs: {unknown_ids}.")
        already_reported_ids = sorted(set(score_ids) & self._buffered_scores.keys())
        if already_reported_ids:
            raise ValueError(
                "Bundled-core candidate IDs were already submitted: " f"{already_reported_ids}."
            )

        newly_buffered = dict(zip(score_ids, scores, strict=True))
        objective_values = {}
        for candidate_id, score in newly_buffered.items():
            value = score.norm_score if self._search_config.normalize else score.score
            if value is None:
                raise ValueError("A normalized bundled-core score is missing its value.")
            objective_values[candidate_id] = _objective_values(
                value, self._search_config.num_objectives
            )

        prospective_scores = {**self._buffered_scores, **newly_buffered}
        if prospective_scores.keys() != pending_ids:
            self._buffered_scores.update(newly_buffered)
            return

        response = ResponseTemplate(evaluated_params=[])
        assert isinstance(response.evaluated_params, list)
        for candidate_id in self._pending_ids:
            if candidate_id not in objective_values:
                score = prospective_scores[candidate_id]
                value = score.norm_score if self._search_config.normalize else score.score
                assert value is not None
                objective_values[candidate_id] = _objective_values(
                    value, self._search_config.num_objectives
                )
            response.evaluated_params.append(
                EvaluatedParamResponse(
                    id=self._core_ids_by_pending_id[candidate_id],
                    scores=objective_values[candidate_id],
                )
            )
        assert self._core_socket is not None
        self._core_ipc.send_to_core(self._core_socket, response)
        self._buffered_scores.clear()
        self._pending_ids = ()
        self._pending_candidates.clear()
        self._core_ids_by_pending_id.clear()

    def close(self) -> None:
        """Stop the process and close both sides of adapter-owned IPC."""

        self._core_ipc.stop()
        if self._core_socket is not None:
            self._core_socket.close()
            self._core_socket = None
        if self._listen_socket is not None:
            self._listen_socket.close()
            self._listen_socket = None
        self._started = False
