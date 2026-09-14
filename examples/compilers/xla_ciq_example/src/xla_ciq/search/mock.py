"""Deterministic mocked CompileIQ search (--mock / CI)."""

from __future__ import annotations

from xla_ciq.search.types import SearchRequest, SearchResult

MOCK_BANNER = (
    "MOCK: CompileIQ search is mocked. "
    "Output ACF bytes are synthetic and not production controls."
)


class MockSearchEngine:
    def run(self, request: SearchRequest) -> SearchResult:
        notes = [
            MOCK_BANNER,
            f"MOCK: kernel={request.kernel_name}",
            f"MOCK: shape={request.shape}",
            f"MOCK: arch={request.arch} cuda_version={request.cuda_version}",
        ]

        header = b"MOCK-ACF-v1\n"
        body = (
            f"kernel={request.kernel_name}\n"
            f"arch={request.arch}\n"
            f"cuda_version={request.cuda_version}\n"
            f"shape={request.shape}\n"
            f"source_sha256_prefix={_sha_prefix(request.source_bytes)}\n"
        ).encode("utf-8")
        return SearchResult(acf_bytes=header + body, notes=notes, score=None)


def _sha_prefix(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()[:16]
