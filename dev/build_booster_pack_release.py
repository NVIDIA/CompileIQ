#!/usr/bin/env python3
"""Build auditable Booster Pack release assets from approved local inputs."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import re
import shutil
import sys
import zipfile
from collections.abc import Mapping, Sequence


DEFAULT_DOCS_URL = "https://nvidia.github.io/CompileIQ/stable/booster_packs.html"
PRIOR_RELEASE_CATALOG_NAME = ".booster-pack-catalog.prior-release.json"
TAG_RE = re.compile(r"^booster-packs-(\d{4})\.(\d{2})\.(\d{2})(?:[-.][A-Za-z0-9._-]+)?$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2) + "\n").encode("utf-8")


def _mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{context}: expected object")
    return value


def _list(value: object, context: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{context}: expected list")
    return value


def _today_tag() -> str:
    return f"booster-packs-{dt.date.today().strftime('%Y.%m.%d')}"


def _date_parts_from_tag(tag: str) -> tuple[str, str]:
    match = TAG_RE.fullmatch(tag)
    if not match:
        raise ValueError("tag must match booster-packs-YYYY.MM.DD[-suffix]")
    year, month, day = match.groups()
    return f"{year}.{month}.{day}", f"{year}-{month}-{day}T00:00:00Z"


def _find_manifest_path(zip_path: pathlib.Path) -> str:
    with zipfile.ZipFile(zip_path) as archive:
        manifest_paths = [
            name
            for name in archive.namelist()
            if pathlib.PurePosixPath(name).name == "booster-pack-manifest.json"
        ]

    if not manifest_paths:
        raise ValueError(f"{zip_path.name}: missing booster-pack-manifest.json")
    if len(manifest_paths) > 1:
        raise ValueError(
            f"{zip_path.name}: expected one booster-pack-manifest.json, found "
            + ", ".join(sorted(manifest_paths))
        )
    return manifest_paths[0]


def _compact_acf_entries(manifest: Mapping[str, object]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for index, entry in enumerate(_list(manifest.get("acfs"), "manifest acfs")):
        acf = _mapping(entry, f"manifest acfs[{index}]")
        result.append(
            {
                "filename": acf.get("filename"),
                "sha256": acf.get("sha256"),
                "size_bytes": acf.get("size_bytes"),
            }
        )
    return result


def _rewrite_pack_zip(
    source_zip: pathlib.Path,
    output_zip: pathlib.Path,
    tag: str,
    catalog_version: str,
) -> tuple[str, str, dict[str, object]]:
    manifest_path = _find_manifest_path(source_zip)
    manifest_digest = ""
    rewritten_manifest: dict[str, object] = {}

    with (
        zipfile.ZipFile(source_zip, "r") as zin,
        zipfile.ZipFile(
            output_zip,
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as zout,
    ):
        for info in zin.infolist():
            payload = zin.read(info.filename)
            if info.filename == manifest_path:
                rewritten_manifest = _mapping(json.loads(payload), f"{source_zip.name}: manifest")
                rewritten_manifest["catalog_version"] = catalog_version
                rewritten_manifest["release_tag"] = tag
                payload = json_bytes(rewritten_manifest)
                manifest_digest = sha256_bytes(payload)

            out_info = zipfile.ZipInfo(info.filename, ZIP_TIMESTAMP)
            out_info.external_attr = info.external_attr
            out_info.compress_type = zipfile.ZIP_DEFLATED
            zout.writestr(out_info, payload)

    if not manifest_digest:
        raise ValueError(f"{source_zip.name}: manifest was not rewritten")
    return manifest_path, manifest_digest, rewritten_manifest


def _render_list(value: object) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    if value is None:
        return ""
    return str(value)


def _validation_evidence(pack: Mapping[str, object]) -> str:
    validation = pack.get("validation_summary")
    if isinstance(validation, Mapping):
        return _render_list(validation.get("evidence"))
    return _render_list(validation)


def _pack_label(pack: Mapping[str, object]) -> str:
    artifact_name = _render_list(pack.get("artifact_name"))
    display_name = _render_list(pack.get("display_name")) or _render_list(pack.get("pack_id"))
    if display_name and artifact_name:
        return f"{display_name} (`{artifact_name}`)"
    if artifact_name:
        return f"`{artifact_name}`"
    return display_name or "Unknown Booster Pack"


def _pack_entries_by_artifact(catalog: Mapping[str, object]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for index, pack in enumerate(_list(catalog.get("packs"), "catalog packs")):
        pack_obj = _mapping(pack, f"catalog packs[{index}]")
        artifact_name = pack_obj.get("artifact_name")
        if not isinstance(artifact_name, str) or not artifact_name:
            raise ValueError(f"catalog packs[{index}]: missing artifact_name")
        result[artifact_name] = pack_obj
    return result


def _load_prior_release_catalog(source_dir: pathlib.Path) -> dict[str, object] | None:
    prior_path = source_dir / PRIOR_RELEASE_CATALOG_NAME
    if not prior_path.is_file():
        return None
    return _mapping(json.loads(prior_path.read_text()), PRIOR_RELEASE_CATALOG_NAME)


def _release_changes(
    source_dir: pathlib.Path,
    catalog: Mapping[str, object],
    prior_catalog: Mapping[str, object] | None,
) -> dict[str, list[str]]:
    current = _pack_entries_by_artifact(catalog)
    if prior_catalog is None:
        return {
            "added": [_pack_label(current[name]) for name in sorted(current)],
            "updated": [],
            "removed": [],
        }

    prior = _pack_entries_by_artifact(prior_catalog)
    added = [_pack_label(current[name]) for name in sorted(set(current) - set(prior))]
    removed = [_pack_label(prior[name]) for name in sorted(set(prior) - set(current))]
    updated: list[str] = []
    for artifact_name in sorted(set(current) & set(prior)):
        prior_sha = prior[artifact_name].get("artifact_sha256")
        if (
            isinstance(prior_sha, str)
            and SHA256_RE.fullmatch(prior_sha)
            and sha256_file(source_dir / artifact_name) != prior_sha
        ):
            updated.append(_pack_label(current[artifact_name]))

    return {"added": added, "updated": updated, "removed": removed}


def _render_change_section(title: str, labels: Sequence[str]) -> list[str]:
    if not labels:
        return []
    lines = [f"### {title}", ""]
    lines.extend(f"- {label}" for label in labels)
    lines.append("")
    return lines


def _render_pack_section(pack: Mapping[str, object]) -> list[str]:
    acfs = pack.get("acfs")
    acf_count = len(acfs) if isinstance(acfs, list) else _render_list(pack.get("acf_count"))
    lines = [
        f"### {_render_list(pack.get('display_name'))}",
        "",
    ]

    description = _render_list(pack.get("description"))
    if description:
        lines.extend([description, ""])

    lines.extend(
        [
            f"- Asset: `{_render_list(pack.get('artifact_name'))}`",
            f"- Pack id: `{_render_list(pack.get('pack_id'))}`",
            f"- Pack type: `{_render_list(pack.get('pack_type'))}`",
            f"- CUDA Toolkit: `{_render_list(pack.get('cuda_version'))}`",
            f"- Controls stage: `{_render_list(pack.get('controls_stage'))}`",
            f"- Supported GPUs: `{_render_list(pack.get('supported_gpus'))}`",
            f"- ACF count: `{acf_count}`",
        ]
    )
    validation = _validation_evidence(pack)
    if validation:
        lines.append(f"- Validation evidence: `{validation}`")
    lines.append("")
    return lines


def _render_release_body(
    catalog: Mapping[str, object],
    docs_url: str,
    changes: Mapping[str, Sequence[str]],
) -> str:
    packs = [
        _mapping(pack, "catalog pack") for pack in _list(catalog.get("packs"), "catalog packs")
    ]
    catalog_date = _render_list(catalog.get("catalog_version")) or _render_list(
        catalog.get("release_tag")
    )
    lines = [
        "## Booster Pack Catalog Release",
        "",
        f"Documentation: {docs_url}",
        "",
        f"This release contains the complete Booster Pack catalog as of {catalog_date}.",
        "",
        "Assets:",
        "",
        "- `booster-pack-catalog.json`",
    ]
    lines.extend(f"- `{_render_list(pack.get('artifact_name'))}`" for pack in packs)
    lines.extend(["- `SHA256SUMS.txt`", ""])

    for pack in packs:
        lines.extend(_render_pack_section(pack))

    change_lines = (
        _render_change_section("Added", changes.get("added", []))
        + _render_change_section("Updated", changes.get("updated", []))
        + _render_change_section("Removed", changes.get("removed", []))
    )
    lines.extend(["## Changes", ""])
    if change_lines:
        lines.extend(change_lines)
    else:
        lines.extend(["No added, removed, or updated packs.", ""])

    return "\n".join(lines).rstrip() + "\n"


def build_release(
    source_dir: pathlib.Path,
    output_dir: pathlib.Path,
    tag: str,
    catalog_version: str | None,
    generated_at: str | None,
    docs_url: str,
    clean_output: bool,
) -> Sequence[pathlib.Path]:
    if not source_dir.is_dir():
        raise ValueError(f"source directory does not exist: {source_dir}")

    default_catalog_version, default_generated_at = _date_parts_from_tag(tag)
    catalog_version = catalog_version or default_catalog_version
    generated_at = generated_at or default_generated_at

    if output_dir.resolve() == source_dir.resolve():
        raise ValueError("output directory must differ from source directory")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not clean_output:
            raise ValueError(f"output directory is not empty: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    catalog_path = source_dir / "booster-pack-catalog.json"
    catalog = _mapping(json.loads(catalog_path.read_text()), "booster-pack-catalog.json")
    prior_catalog = _load_prior_release_catalog(source_dir)
    catalog["catalog_version"] = catalog_version
    catalog["generated_at"] = generated_at
    catalog["release_tag"] = tag

    output_assets: list[pathlib.Path] = []
    for index, pack in enumerate(_list(catalog.get("packs"), "catalog packs")):
        pack_obj = _mapping(pack, f"catalog packs[{index}]")
        artifact_name = pack_obj.get("artifact_name")
        if not isinstance(artifact_name, str) or not artifact_name:
            raise ValueError(f"catalog packs[{index}]: missing artifact_name")

        source_zip = source_dir / artifact_name
        output_zip = output_dir / artifact_name
        if not source_zip.is_file():
            raise ValueError(f"missing pack zip named by catalog: {source_zip}")

        manifest_path, manifest_digest, manifest = _rewrite_pack_zip(
            source_zip,
            output_zip,
            tag,
            catalog_version,
        )
        output_assets.append(output_zip)

        pack_obj["artifact_sha256"] = sha256_file(output_zip)
        pack_obj["artifact_size_bytes"] = output_zip.stat().st_size
        pack_obj["manifest_path"] = manifest_path
        pack_obj["manifest_sha256"] = manifest_digest
        pack_obj["acf_count"] = len(_list(manifest.get("acfs"), f"{artifact_name}: manifest acfs"))
        pack_obj["acfs"] = _compact_acf_entries(manifest)

        for field in (
            "display_name",
            "pack_id",
            "pack_type",
            "description",
            "cuda_version",
            "controls_stage",
            "supported_gpus",
            "validation_summary",
        ):
            if field in manifest:
                pack_obj[field] = manifest[field]

    output_catalog = output_dir / "booster-pack-catalog.json"
    output_catalog.write_bytes(json_bytes(catalog))

    release_body = output_dir / "release-body.md"
    changes = _release_changes(source_dir, catalog, prior_catalog)
    release_body.write_text(_render_release_body(catalog, docs_url, changes))

    checksum_targets = [output_catalog, *output_assets]
    checksums = [f"{sha256_file(path)}  {path.name}" for path in checksum_targets]
    checksum_path = output_dir / "SHA256SUMS.txt"
    checksum_path.write_text("\n".join(checksums) + "\n")

    return [output_catalog, *output_assets, checksum_path, release_body]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=pathlib.Path,
        required=True,
        help="Directory containing an approved booster-pack-catalog.json and pack zips.",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        help="Clean output directory. Defaults to dist/booster-pack-release/<tag>.",
    )
    parser.add_argument(
        "--tag",
        default=_today_tag(),
        help=(
            "Release tag. Defaults to booster-packs-YYYY.MM.DD for today; "
            "repair releases may use booster-packs-YYYY.MM.DD-rev1."
        ),
    )
    parser.add_argument(
        "--catalog-version",
        help="Catalog version written into catalog and manifests. Defaults to the tag date.",
    )
    parser.add_argument(
        "--generated-at",
        help=(
            "Catalog generated_at timestamp. Defaults to the deterministic tag date "
            "at 00:00:00Z."
        ),
    )
    parser.add_argument(
        "--docs-url",
        default=DEFAULT_DOCS_URL,
        help=f"Public docs URL written into release-body.md. Default: {DEFAULT_DOCS_URL}",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete the output directory before writing generated assets.",
    )
    args = parser.parse_args(argv)

    output_dir = args.output_dir or pathlib.Path("dist") / "booster-pack-release" / args.tag
    try:
        assets = build_release(
            args.source_dir,
            output_dir,
            args.tag,
            args.catalog_version,
            args.generated_at,
            args.docs_url,
            args.clean_output,
        )
    except (OSError, ValueError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote Booster Pack release assets for {args.tag} to {output_dir}.")
    print("Generated files:")
    for asset in assets:
        print(f"- {asset.name}")
    print("Upload the catalog, zip assets, and SHA256SUMS.txt. Use release-body.md as notes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
