"""Shared pytest helpers."""

from __future__ import annotations


def populate_argv(*args: str) -> list[str]:
    """Build populate argv with --mock for deterministic unit tests."""
    return ["bundle", "populate", *args, "--mock"]
