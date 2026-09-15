"""Search engine interface."""

from __future__ import annotations

from typing import Protocol

from xla_ciq.search.types import SearchRequest, SearchResult


class SearchEngine(Protocol):
    def run(self, request: SearchRequest) -> SearchResult:
        """Run a search for one kernel/shape and return ACF bytes."""
