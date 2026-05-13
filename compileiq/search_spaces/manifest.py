"""
Manifest models for the search-space release catalog.

A ``manifest.json`` published alongside each release acts like a table of
contents for the release: it describes every search-space binary in that
catalog. The resolver consults the manifest to map a logical request
``(compiler, compiler_version, variant)`` to a concrete asset.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

MANIFEST_JSON_SCHEMA_ID = (
    "https://raw.githubusercontent.com/NVIDIA/CompileIQ/main/"
    "schemas/search-space-manifest-v1.schema.json"
)
"""Public JSON Schema identifier for the release manifest validation contract."""


def validate_asset_filename(filename: str) -> str:
    """Validate a release asset filename and return it unchanged.

    Manifest filenames are GitHub release asset basenames, not filesystem paths.
    Keep the contract narrow so local mirrors and caches cannot escape their
    intended directories through absolute paths or parent traversal.
    """
    if not isinstance(filename, str):
        raise ValueError("filename must be a string")
    if not filename:
        raise ValueError("filename must not be empty")
    if "/" in filename or "\\" in filename or ":" in filename:
        raise ValueError("filename must be a plain asset basename")
    if filename in {".", ".."}:
        raise ValueError("filename must not contain parent traversal")
    return filename


class SearchSpaceEntry(BaseModel, populate_by_name=True, extra="forbid"):
    """One entry in the manifest, describing a single search-space binary."""

    compiler: Literal["ptxas", "nvcc"]
    """Compiler family this search space is built for."""

    compiler_version: str
    """Compiler version this search space targets, e.g. ``"13.3"``."""

    variant: str = "default"
    """Variant identifier within (compiler, compiler_version), e.g. ``"att"``,
    ``"p0"``, ``"p2"``. Use ``"default"`` for the canonical variant."""

    filename: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    """Asset filename in the GitHub release."""

    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    """Lowercase hex sha256 of the asset, verified on download and cache hit."""

    size_bytes: int = Field(gt=0)
    """Asset size in bytes, used as a quick sanity check before hashing."""

    search_space_format: str = "1.0.0"
    """Version of the search-space binary's internal schema."""

    description: str | None = None
    """Optional human-readable description shown in error messages and tooling."""

    @field_validator("filename")
    @classmethod
    def reject_path_like_filename(cls, value: str) -> str:
        return validate_asset_filename(value)


class SearchSpaceManifestModel(BaseModel, populate_by_name=True, extra="forbid"):
    """Top-level release catalog published as ``manifest.json``."""

    manifest_format: Literal["1.0.0"] = "1.0.0"
    """Version of the manifest/catalog schema, not the binary file format."""

    tag: str
    """The release tag this manifest belongs to. Used to validate the manifest
    matches the release the resolver was directed at."""

    generated_at: str
    """ISO-8601 timestamp of when this manifest was generated. Informational only."""

    entries: list[SearchSpaceEntry]
    """All search-space binaries available in this release."""

    @model_validator(mode="after")
    def reject_duplicate_selectors(self) -> "SearchSpaceManifestModel":
        """Prevent ambiguous lookups for the public resolver key."""
        seen: set[tuple[str, str, str]] = set()
        for entry in self.entries:
            key = (entry.compiler, entry.compiler_version, entry.variant)
            if key in seen:
                raise ValueError(
                    "Duplicate manifest entry for "
                    f"compiler={entry.compiler} "
                    f"compiler_version={entry.compiler_version} "
                    f"variant={entry.variant}"
                )
            seen.add(key)
        return self

    def find(
        self,
        compiler: str,
        compiler_version: str,
        variant: str = "default",
    ) -> SearchSpaceEntry:
        """Return the manifest entry matching the request."""
        for entry in self.entries:
            if (
                entry.compiler == compiler
                and entry.compiler_version == compiler_version
                and entry.variant == variant
            ):
                return entry

        available = (
            ", ".join(f"{e.compiler}/{e.compiler_version}/{e.variant}" for e in self.entries)
            or "(empty manifest)"
        )
        raise LookupError(
            f"No manifest entry for compiler={compiler} version={compiler_version} "
            f"variant={variant}. Available: {available}"
        )


def search_space_manifest_json_schema() -> dict[str, object]:
    """Return the public JSON Schema for ``manifest_format == "1.0.0"``.

    The Pydantic model remains the runtime source of truth. The checked-in schema
    generated from this helper gives release tooling and reviewers a stable
    contract to inspect.
    """
    schema = SearchSpaceManifestModel.model_json_schema()
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": MANIFEST_JSON_SCHEMA_ID,
        **schema,
    }
