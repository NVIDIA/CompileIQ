#!/usr/bin/env python3
"""Deploy versioned CompileIQ Sphinx docs to a local gh-pages branch.

The caller builds docs for one ref, runs this script, then pushes gh-pages.
This script never pushes to GitHub.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
from dataclasses import dataclass
from pathlib import Path


BASE_URL = "https://nvidia.github.io/CompileIQ"
DEFAULT_HTML_DIR = Path("public/html")
DEFAULT_PAGES_REF = "gh-pages"

MAIN_REF = "refs/heads/main"
RELEASE_BRANCH_RE = re.compile(r"^refs/heads/release-(\d+\.\d+)$")
FINAL_TAG_RE = re.compile(r"^refs/tags/v(\d+\.\d+)\.\d+$")
CATALOG_TAG_RE = re.compile(r"^refs/tags/(booster-packs|search-spaces)-")
VERSION_RE = re.compile(r"^\d+\.\d+$")

DEPLOY_EXCLUDE = shutil.ignore_patterns(".doctrees", "__pycache__", "*.pyc")


@dataclass(frozen=True)
class DeploymentPlan:
    version: str
    metadata_only: bool
    skip: bool = False
    reason: str = ""


def git_run_cmd(*args: str, cwd: Path | None = None) -> str:
    """Run a git command, printing stderr before raising on failure."""

    cmd = ("git", *args)
    print(f"  $ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=cwd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        if result.stdout:
            print(result.stdout, flush=True)
        if result.stderr:
            print(result.stderr, file=sys.stderr, flush=True)
        result.check_returncode()
    return result.stdout.strip()


def resolve_version_folder(version: str) -> str:
    if version == "latest":
        return "latest"
    if not VERSION_RE.fullmatch(version):
        raise ValueError(
            f"Invalid version '{version}': expected 'latest' or MAJOR.MINOR, e.g. '1.0'"
        )
    return f"v{version}"


def branch_exists(repo_root: Path, branch: str) -> bool:
    return bool(git_run_cmd("branch", "--list", branch, cwd=repo_root))


def remote_branch_exists(repo_root: Path, branch: str, remote: str = "origin") -> bool:
    result = subprocess.run(
        ("git", "ls-remote", "--heads", remote, branch),
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def pages_folder_exists(repo_root: Path, pages_ref: str, folder: str) -> bool:
    result = subprocess.run(
        ("git", "ls-tree", "--name-only", pages_ref),
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return False
    return folder in result.stdout.splitlines()


def determine_deployment(github_ref: str, repo_root: Path, pages_ref: str) -> DeploymentPlan:
    if github_ref == MAIN_REF:
        return DeploymentPlan(version="latest", metadata_only=False)

    release_match = RELEASE_BRANCH_RE.fullmatch(github_ref)
    if release_match:
        return DeploymentPlan(version=release_match.group(1), metadata_only=False)

    final_tag_match = FINAL_TAG_RE.fullmatch(github_ref)
    if final_tag_match:
        version = final_tag_match.group(1)
        folder = resolve_version_folder(version)
        return DeploymentPlan(
            version=version,
            metadata_only=pages_folder_exists(repo_root, pages_ref, folder),
        )

    if github_ref.startswith("refs/tags/v"):
        return DeploymentPlan(
            version="",
            metadata_only=False,
            skip=True,
            reason="non-final package tags do not promote /stable/",
        )

    if CATALOG_TAG_RE.match(github_ref):
        return DeploymentPlan(
            version="",
            metadata_only=False,
            skip=True,
            reason="catalog release tags do not deploy package docs",
        )

    raise ValueError(f"Unsupported docs deployment ref: {github_ref}")


def discover_versions(gh_pages_dir: Path) -> list[str]:
    versions: list[str] = []
    for entry in gh_pages_dir.iterdir():
        if not entry.is_dir():
            continue
        match = re.fullmatch(r"v(\d+\.\d+)", entry.name)
        if match:
            versions.append(match.group(1))
    versions.sort(key=lambda item: tuple(int(part) for part in item.split(".")), reverse=True)
    return versions


def is_released(repo_root: Path, version: str) -> bool:
    result = subprocess.run(
        ("git", "tag", "-l", f"v{version}.*"),
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    pattern = re.compile(rf"^v{re.escape(version)}\.\d+$")
    return any(pattern.fullmatch(tag) for tag in result.stdout.splitlines())


def generate_versions_json(
    gh_pages_dir: Path,
    *,
    versions: list[str],
    released: list[str],
    has_latest: bool,
    base_url: str,
) -> None:
    entries: list[dict[str, object]] = []

    if has_latest:
        entries.append(
            {
                "name": "latest (main)",
                "version": "latest",
                "url": f"{base_url}/latest/",
            }
        )

    released_set = set(released)
    preferred = released[0] if released else None

    for version in versions:
        entry: dict[str, object] = {
            "version": version,
            "url": f"{base_url}/v{version}/",
        }
        if version == preferred:
            entry["name"] = f"{version} (stable)"
            entry["url"] = f"{base_url}/stable/"
            entry["preferred"] = True
        elif version not in released_set:
            entry["name"] = f"{version} (prerelease)"
        else:
            entry["name"] = version
        entries.append(entry)

    path = gh_pages_dir / "versions.json"
    path.write_text(json.dumps(entries, indent=2) + "\n")
    print(f"Wrote {path} with {len(entries)} entries.")


def generate_root_redirect(gh_pages_dir: Path, target: str) -> None:
    html = f"""\
<!DOCTYPE html>
<html>
<head>
  <meta http-equiv="refresh" content="0; url={target}" />
  <script>window.location.href = "{target}";</script>
</head>
<body>
  <p>Redirecting to <a href="{target}">{target}</a>...</p>
</body>
</html>
"""
    (gh_pages_dir / "index.html").write_text(html)
    print(f"Root redirect -> {target}")


def generate_404_redirect(gh_pages_dir: Path, target: str, base_url: str) -> None:
    prefix = urllib.parse.urlparse(base_url).path.rstrip("/") + "/"
    html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta http-equiv="X-UA-Compatible" content="IE=edge,chrome=1" />
  <meta name="viewport" content="width=device-width" />
  <title>Page not found - CompileIQ</title>
  <script>
(function () {{
  var prefix = {json.dumps(prefix)};
  var path = location.pathname;
  if (path.indexOf(prefix) === 0) {{
    path = path.slice(prefix.length);
  }}
  if (/^(stable|latest|v\\d+\\.\\d+)(\\/|$)/.test(path)) {{
    return;
  }}
  location.replace(prefix + {json.dumps(target)} + path + location.search + location.hash);
}})();
  </script>
  <style>
    body {{
      background: #f1f1f1;
      color: #222;
      font: 14px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
      margin: 0;
    }}
    .container {{
      margin: 0 auto;
      max-width: 600px;
      padding: 20px 0 40px;
      text-align: center;
    }}
    h1 {{
      color: #222;
      font-size: 144px;
      font-weight: 800;
      letter-spacing: 0;
      line-height: 1;
      margin: 0;
    }}
    h2 {{
      color: #5a5a5a;
      font-size: 24px;
      font-weight: 400;
      margin: 12px 0 24px;
    }}
    p {{
      color: #5a5a5a;
      font-size: 14px;
      margin: 0.5em 0;
    }}
    a {{ color: #76b900; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>404</h1>
    <h2>File not found.</h2>
    <p>The page you requested does not exist in this version of the CompileIQ documentation.</p>
    <p>
      <a href="{prefix}stable/">Latest stable docs</a>
      &middot;
      <a href="{prefix}latest/">Development docs</a>
    </p>
  </div>
</body>
</html>
"""
    (gh_pages_dir / "404.html").write_text(html)
    print(f"404 fallback -> {target}")


def update_stable(gh_pages_dir: Path, stable_version: str) -> None:
    stable_dir = gh_pages_dir / "stable"
    source_dir = gh_pages_dir / f"v{stable_version}"

    if stable_dir.exists():
        shutil.rmtree(stable_dir)
    shutil.copytree(source_dir, stable_dir)
    print(f"/stable/ -> v{stable_version}")


def deploy_docs(
    *,
    version: str,
    metadata_only: bool,
    repo_root: Path,
    html_dir: Path,
    pages_ref: str,
    base_url: str,
) -> None:
    folder = resolve_version_folder(version)
    repo_root = repo_root.resolve()
    html_dir = (repo_root / html_dir).resolve() if not html_dir.is_absolute() else html_dir
    base_url = base_url.rstrip("/")

    print(f"Deploying docs to /{folder}/ (metadata_only={metadata_only})")

    if not metadata_only and not html_dir.exists():
        raise FileNotFoundError(f"Built docs not found at {html_dir}")

    has_local_branch = branch_exists(repo_root, pages_ref)
    if not has_local_branch and remote_branch_exists(repo_root, pages_ref):
        raise RuntimeError(
            f"origin has a {pages_ref} branch but it is not present locally. Fetch first:\n"
            f"  git fetch origin {pages_ref}:{pages_ref}"
        )

    with tempfile.TemporaryDirectory() as tmp:
        worktree = Path(tmp) / pages_ref
        if has_local_branch:
            git_run_cmd("worktree", "add", str(worktree), pages_ref, cwd=repo_root)
        else:
            git_run_cmd("worktree", "add", "--detach", str(worktree), "HEAD", cwd=repo_root)
            git_run_cmd("checkout", "--orphan", pages_ref, cwd=worktree)
            git_run_cmd("rm", "-rf", ".", cwd=worktree)

        try:
            target = worktree / folder
            if metadata_only:
                if not target.exists():
                    raise RuntimeError(
                        f"--metadata-only requires /{folder}/ to exist on {pages_ref}"
                    )
            else:
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(html_dir, target, ignore=DEPLOY_EXCLUDE)
                print(f"Copied {html_dir} -> {target}")

            versions = discover_versions(worktree)
            released = [item for item in versions if is_released(repo_root, item)]
            has_latest = (worktree / "latest").is_dir()
            print(
                f"Deployed versions: {versions}; released: {released}; "
                f"has latest: {has_latest}"
            )

            if released:
                update_stable(worktree, released[0])
                generate_root_redirect(worktree, "stable/")
                generate_404_redirect(worktree, "stable/", base_url)
            else:
                stable_dir = worktree / "stable"
                if stable_dir.exists():
                    shutil.rmtree(stable_dir)
                    print("Removed stale /stable/ because no final release tag exists.")
                if has_latest:
                    generate_root_redirect(worktree, "latest/")
                    generate_404_redirect(worktree, "latest/", base_url)

            generate_versions_json(
                worktree,
                versions=versions,
                released=released,
                has_latest=has_latest,
                base_url=base_url,
            )
            (worktree / ".nojekyll").touch()

            git_run_cmd("add", "-A", cwd=worktree)
            if not git_run_cmd("status", "--porcelain", cwd=worktree):
                print("No docs deployment changes to commit.")
                return

            git_run_cmd(
                "-c",
                "user.email=actions@github.com",
                "-c",
                "user.name=GitHub Actions",
                "-c",
                "commit.gpgsign=false",
                "-c",
                "core.hooksPath=/dev/null",
                "commit",
                "-m",
                f"Deploy docs: {folder}",
                cwd=worktree,
            )
        finally:
            git_run_cmd("worktree", "remove", "--force", str(worktree), cwd=repo_root)


def write_github_output(path: Path, values: dict[str, str]) -> None:
    with path.open("a", encoding="utf-8") as output:
        for key, value in values.items():
            output.write(f"{key}={value}\n")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="Determine docs deployment mode")
    plan.add_argument("--github-ref", required=True)
    plan.add_argument("--repo-root", type=Path, default=Path("."))
    plan.add_argument("--pages-ref", default=DEFAULT_PAGES_REF)
    plan.add_argument("--github-output", type=Path)

    deploy = subparsers.add_parser("deploy", help="Deploy built docs to gh-pages")
    deploy.add_argument("--version", required=True)
    deploy.add_argument("--metadata-only", action="store_true")
    deploy.add_argument("--repo-root", type=Path, default=Path("."))
    deploy.add_argument("--html-dir", type=Path, default=DEFAULT_HTML_DIR)
    deploy.add_argument("--pages-ref", default=DEFAULT_PAGES_REF)
    deploy.add_argument("--base-url", default=BASE_URL)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    try:
        if args.command == "plan":
            plan = determine_deployment(args.github_ref, args.repo_root, args.pages_ref)
            values = {
                "version": plan.version,
                "metadata_only": str(plan.metadata_only).lower(),
                "skip": str(plan.skip).lower(),
                "reason": plan.reason,
            }
            print(
                "Docs deployment plan: "
                f"version={plan.version or '<none>'} "
                f"metadata_only={plan.metadata_only} skip={plan.skip} {plan.reason}"
            )
            if args.github_output:
                write_github_output(args.github_output, values)
            return 0

        if args.command == "deploy":
            deploy_docs(
                version=args.version,
                metadata_only=args.metadata_only,
                repo_root=args.repo_root,
                html_dir=args.html_dir,
                pages_ref=args.pages_ref,
                base_url=args.base_url,
            )
            return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
