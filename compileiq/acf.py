"""
Translate ACFs between CUDA Toolkit (CTK) versions.

An ACF only works with the CTK release it was tuned for. Translation re-targets an
existing ACF to another CTK's search spaces. Each control is matched by its key path, and
the target search space's type for it decides the result: range and enum controls keep
their value when the target allows it and are dropped otherwise, and literals take the
target's value. Controls the target doesn't have are dropped, and the version metadata is
rewritten for the target CTK. Decryption and validation happen inside ``libciq``.

A translated ACF carries no guarantee that the original's performance carries over;
benchmark it before shipping it.

Usage::

    python -m compileiq.acf translate kernel.acf --to 13.4 -o kernel.13.4.acf

Search spaces are resolved like ``PtxasSearchSpace`` and ``NvccSearchSpace``, so
``CIQ_SEARCH_SPACES_DIR`` makes translation work offline.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
from typing import Sequence

from compileiq.search_spaces.resolver import resolve
from compileiq.utils._libciq import (
    AcfTranslationError,
    LibciqUnavailableError,
    translate_controls,
)
from compileiq.utils.helpers import load_compiler_config, save_compiler_config

__all__ = [
    "AcfTranslationError",
    "LibciqUnavailableError",
    "translate_acf_bytes",
    "translate_compiler_config",
]


def translate_acf_bytes(acf: bytes, search_space_paths: Sequence[str | os.PathLike]) -> bytes:
    """Translate encrypted ACF bytes against explicit target search space files.

    ``search_space_paths`` holds at most one search space per compiler; sections of the
    ACF are paired with the search space for the same compiler.
    """
    return translate_controls(acf, search_space_paths)


def _target_search_spaces(version: str, ptxas_variant: str, tag: str) -> list[pathlib.Path]:
    """Resolve every published compiler search space for the target CTK.

    The caller can't see which compiler sections an encrypted ACF contains, so every
    compiler's search space is passed and libciq picks the ones it needs.
    """
    requests = [("ptxas", ptxas_variant), ("nvcc", "default")]
    paths = []
    for compiler, variant in requests:
        try:
            paths.append(resolve(compiler, version, variant, tag))
        except LookupError:
            if compiler == "ptxas" and variant != "default":
                raise
    if not paths:
        raise LookupError(f"No search spaces are published for CTK {version}.")
    return paths


def translate_compiler_config(
    acf_hex: str,
    version: str,
    ptxas_variant: str = "default",
    tag: str = "latest",
) -> str:
    """Translate an ACF to another CTK version.

    Args:
        acf_hex: The ACF as a hex string, as returned by ``load_compiler_config``.
        version: The target CTK version, for example ``"13.4"``.
        ptxas_variant: The target PTXAS search-space variant. ACFs don't record the
            variant they were tuned on; controls the target variant doesn't contain
            are dropped.
        tag: The search-space release to use, as for ``PtxasSearchSpace``.

    Returns:
        The translated ACF as a hex string, ready for ``save_compiler_config``.

    Raises:
        AcfTranslationError: The ACF can't be translated.
        LibciqUnavailableError: This installation has no libciq that can translate.
        LookupError: No search spaces are published for the target CTK or variant.
    """
    paths = _target_search_spaces(version, ptxas_variant, tag)
    return translate_acf_bytes(bytes.fromhex(acf_hex), paths).hex()


def _default_output(acf_path: pathlib.Path, version: str) -> pathlib.Path:
    return acf_path.with_name(f"{acf_path.stem}.{version}{acf_path.suffix or '.acf'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m compileiq.acf",
        description="Work with Advanced Controls Files (ACFs).",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    translate = commands.add_parser(
        "translate",
        help="Translate an ACF to another CUDA Toolkit version.",
        description=(
            "Translate an ACF to another CUDA Toolkit version. Controls the target search "
            "spaces don't allow are dropped. Benchmark the result before shipping it."
        ),
    )
    translate.add_argument("acf", type=pathlib.Path, help="ACF file to translate.")
    translate.add_argument(
        "--to",
        required=True,
        dest="version",
        help="Target CTK version, e.g. 13.4.",
    )
    translate.add_argument(
        "--ptxas-variant",
        default="default",
        help="Target PTXAS search-space variant (default: %(default)s).",
    )
    translate.add_argument(
        "--tag",
        default="latest",
        help="Search-space release tag (default: %(default)s).",
    )
    translate.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        help="Output path (default: <acf stem>.<version>.acf next to the input).",
    )

    args = parser.parse_args(argv)

    output = args.output or _default_output(args.acf, args.version)
    try:
        translated = translate_compiler_config(
            load_compiler_config(str(args.acf)),
            args.version,
            ptxas_variant=args.ptxas_variant,
            tag=args.tag,
        )
    except (LibciqUnavailableError, LookupError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    save_compiler_config(str(output), translated)
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
