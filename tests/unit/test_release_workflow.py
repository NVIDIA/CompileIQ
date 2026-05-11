"""Static checks for search-space release workflow invariants."""

from __future__ import annotations

import pathlib


WORKFLOW = pathlib.Path(__file__).resolve().parents[2] / ".github" / "workflows" / "ci.yml"


def test_search_space_release_workflow_is_not_active_without_artifact_staging():
    content = WORKFLOW.read_text()
    legacy_assets_glob = "/".join(("assets", "*.bin"))

    assert "release-search-spaces:" not in content
    assert legacy_assets_glob not in content
    assert "gh release create" not in content


def test_wheel_release_only_runs_for_version_tags():
    content = WORKFLOW.read_text()
    broad_tag_release_condition = (
        "startsWith(github.ref, 'refs/tags/')\n"
        "    permissions:\n"
        "      contents: write"
    )

    assert 'tags: ["v*"]' in content
    assert 'tags: ["**"]' not in content
    assert 'ref.startswith("refs/tags/v")' in content
    assert '${GITHUB_REF#refs/tags/v}' in content
    assert "startsWith(github.ref, 'refs/tags/v')" in content
    assert broad_tag_release_condition not in content


def test_search_space_release_prep_is_local_until_publish_path_is_decided():
    content = WORKFLOW.read_text()

    assert "startsWith(github.ref, 'refs/tags/search-spaces-')" not in content
