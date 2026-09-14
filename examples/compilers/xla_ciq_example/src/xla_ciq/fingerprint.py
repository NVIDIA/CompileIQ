"""Provisional fingerprint and content-id helpers."""

from __future__ import annotations

import hashlib


def content_id(acf_bytes: bytes) -> str:
    """Return lowercase hex SHA-256 of ACF bytes (DEC-006 provisional)."""
    return hashlib.sha256(acf_bytes).hexdigest()


def fingerprint(
    *,
    cuda_version: str,
    arch: str,
    source_bytes: bytes,
    kernel_name: str,
    shape: str,
) -> str:
    """Exploratory fingerprint used by this example: sha256(cuda|arch|source|kernel|shape).

    Each problem size (shape) gets its own fingerprint so multi-shape
    populate runs produce distinct bundle mappings. A downstream consumer of
    this experimental format would need to reproduce this exact calculation
    from the same canonical inputs. The example does not establish this
    fingerprint as a stable interoperable contract.
    """
    prefix = f"{cuda_version}|{arch}|".encode("utf-8")
    suffix = f"|{kernel_name}|{shape}".encode("utf-8")
    return hashlib.sha256(prefix + source_bytes + suffix).hexdigest()
