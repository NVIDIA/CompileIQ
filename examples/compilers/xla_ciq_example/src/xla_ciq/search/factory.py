"""Factory for search engines."""

from __future__ import annotations

from xla_ciq.search.base import SearchEngine
from xla_ciq.search.compileiq_runtime import CompileIqRuntimeEngine
from xla_ciq.search.mock import MockSearchEngine


def get_search_engine(*, mock: bool) -> SearchEngine:
    if mock:
        return MockSearchEngine()
    return CompileIqRuntimeEngine()
