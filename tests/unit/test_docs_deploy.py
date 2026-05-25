"""Tests for dev/deploy_docs.py."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_DEV = _REPO_ROOT / "dev"
_DEPLOY_SPEC = importlib.util.spec_from_file_location(
    "deploy_docs",
    _DEV / "deploy_docs.py",
)
assert _DEPLOY_SPEC is not None
deploy_docs = importlib.util.module_from_spec(_DEPLOY_SPEC)
sys.modules["deploy_docs"] = deploy_docs
assert _DEPLOY_SPEC.loader is not None
_DEPLOY_SPEC.loader.exec_module(deploy_docs)


def _git(repo: pathlib.Path, *args: str) -> str:
    result = subprocess.run(
        ("git", *args),
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _init_repo(repo: pathlib.Path) -> None:
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    (repo / "README.md").write_text("test repo\n")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "initial")


def _write_html(html_dir: pathlib.Path, title: str) -> None:
    html_dir.mkdir(parents=True, exist_ok=True)
    (html_dir / "index.html").write_text(f"<html><body>{title}</body></html>\n")
    (html_dir / "booster_packs.html").write_text("booster docs\n")
    (html_dir / "compilers_overview.html").write_text("compiler docs\n")
    doctrees = html_dir / ".doctrees"
    doctrees.mkdir()
    (doctrees / "environment.pickle").write_text("not deployed\n")


def _show(repo: pathlib.Path, ref_path: str) -> str:
    return _git(repo, "show", f"gh-pages:{ref_path}")


def test_determine_deployment_modes(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)

    plan = deploy_docs.determine_deployment("refs/heads/main", repo, "gh-pages")
    assert plan.version == "latest"
    assert plan.metadata_only is False
    assert plan.skip is False

    plan = deploy_docs.determine_deployment("refs/heads/release-1.0", repo, "gh-pages")
    assert plan.version == "1.0"
    assert plan.metadata_only is False

    plan = deploy_docs.determine_deployment("refs/tags/v1.0.0", repo, "gh-pages")
    assert plan.version == "1.0"
    assert plan.metadata_only is False

    plan = deploy_docs.determine_deployment("refs/tags/v1.0.0rc1", repo, "gh-pages")
    assert plan.skip is True
    assert "non-final" in plan.reason

    plan = deploy_docs.determine_deployment(
        "refs/tags/search-spaces-2026.05.22",
        repo,
        "gh-pages",
    )
    assert plan.skip is True
    assert "catalog" in plan.reason


def test_determine_deployment_rejects_unsupported_ref(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)

    try:
        deploy_docs.determine_deployment("refs/heads/feature/docs", repo, "gh-pages")
    except ValueError as exc:
        assert "Unsupported docs deployment ref" in str(exc)
    else:
        raise AssertionError("expected unsupported ref to fail")


def test_bootstrap_deploy_latest_creates_gh_pages_tree(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    html_dir = repo / "public" / "html"
    _write_html(html_dir, "latest docs")

    deploy_docs.deploy_docs(
        version="latest",
        metadata_only=False,
        repo_root=repo,
        html_dir=html_dir,
        pages_ref="gh-pages",
        base_url=deploy_docs.BASE_URL,
    )

    assert "latest docs" in _show(repo, "latest/index.html")
    assert "latest/" in _show(repo, "index.html")
    assert "stable/" not in _show(repo, "index.html")
    assert "environment.pickle" not in _git(repo, "ls-tree", "-r", "--name-only", "gh-pages")
    assert _git(repo, "cat-file", "-e", "gh-pages:.nojekyll") == ""

    versions = json.loads(_show(repo, "versions.json"))
    assert versions == [
        {
            "name": "latest (main)",
            "version": "latest",
            "url": "https://nvidia.github.io/CompileIQ/latest/",
        }
    ]


def test_release_branch_and_final_tag_promote_stable(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)

    latest_html = repo / "public" / "latest-html"
    _write_html(latest_html, "latest docs")
    deploy_docs.deploy_docs(
        version="latest",
        metadata_only=False,
        repo_root=repo,
        html_dir=latest_html,
        pages_ref="gh-pages",
        base_url=deploy_docs.BASE_URL,
    )

    release_html = repo / "public" / "release-html"
    _write_html(release_html, "release docs")
    deploy_docs.deploy_docs(
        version="1.0",
        metadata_only=False,
        repo_root=repo,
        html_dir=release_html,
        pages_ref="gh-pages",
        base_url=deploy_docs.BASE_URL,
    )

    assert "release docs" in _show(repo, "v1.0/index.html")
    assert "stable/" not in _git(repo, "ls-tree", "--name-only", "gh-pages")

    _git(repo, "tag", "v1.0.0")
    plan = deploy_docs.determine_deployment("refs/tags/v1.0.0", repo, "gh-pages")
    assert plan.version == "1.0"
    assert plan.metadata_only is True

    deploy_docs.deploy_docs(
        version=plan.version,
        metadata_only=plan.metadata_only,
        repo_root=repo,
        html_dir=repo / "does-not-need-to-exist",
        pages_ref="gh-pages",
        base_url=deploy_docs.BASE_URL,
    )

    assert "release docs" in _show(repo, "stable/index.html")
    assert "stable/" in _show(repo, "index.html")
    assert "/CompileIQ/stable/" in _show(repo, "404.html")
    assert "/CompileIQ/latest/" in _show(repo, "404.html")

    versions = json.loads(_show(repo, "versions.json"))
    assert versions == [
        {
            "name": "latest (main)",
            "version": "latest",
            "url": "https://nvidia.github.io/CompileIQ/latest/",
        },
        {
            "version": "1.0",
            "url": "https://nvidia.github.io/CompileIQ/stable/",
            "name": "1.0 (stable)",
            "preferred": True,
        },
    ]


def test_metadata_only_requires_existing_version(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    html_dir = repo / "public" / "html"
    _write_html(html_dir, "latest docs")
    deploy_docs.deploy_docs(
        version="latest",
        metadata_only=False,
        repo_root=repo,
        html_dir=html_dir,
        pages_ref="gh-pages",
        base_url=deploy_docs.BASE_URL,
    )

    try:
        deploy_docs.deploy_docs(
            version="1.0",
            metadata_only=True,
            repo_root=repo,
            html_dir=repo / "missing",
            pages_ref="gh-pages",
            base_url=deploy_docs.BASE_URL,
        )
    except RuntimeError as exc:
        assert "--metadata-only requires /v1.0/" in str(exc)
    else:
        raise AssertionError("expected metadata-only deploy to fail")


def test_resolve_version_folder_rejects_patch_versions():
    assert deploy_docs.resolve_version_folder("latest") == "latest"
    assert deploy_docs.resolve_version_folder("1.0") == "v1.0"

    try:
        deploy_docs.resolve_version_folder("1.0.0")
    except ValueError as exc:
        assert "MAJOR.MINOR" in str(exc)
    else:
        raise AssertionError("expected patch version to fail")
