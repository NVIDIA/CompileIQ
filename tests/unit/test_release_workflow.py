"""Static checks for search-space release workflow invariants."""

from __future__ import annotations

import pathlib


WORKFLOWS = pathlib.Path(__file__).resolve().parents[2] / ".github" / "workflows"
MAKEFILE = pathlib.Path(__file__).resolve().parents[2] / "Makefile"
CI_WORKFLOW = WORKFLOWS / "ci.yml"
DOCS_WORKFLOW = WORKFLOWS / "docs.yml"


def _make_target_block(content: str, target: str, next_target: str) -> str:
    return content.split(f"{target}:", 1)[1].split(f"\n{next_target}:", 1)[0]


def test_search_space_release_workflow_is_not_active_without_artifact_staging():
    content = CI_WORKFLOW.read_text()
    legacy_assets_glob = "/".join(("assets", "*.bin"))

    assert "release-search-spaces:" not in content
    assert legacy_assets_glob not in content
    assert "gh release create" not in content


def test_wheel_release_only_runs_for_version_tags():
    content = CI_WORKFLOW.read_text()
    broad_tag_release_condition = (
        "startsWith(github.ref, 'refs/tags/')\n"
        "    permissions:\n"
        "      contents: write"
    )

    assert 'tags: ["v*"]' not in content
    assert '"v[0-9]*.[0-9]*.[0-9]*"' in content
    assert 'tags: ["**"]' not in content
    assert 'ref.startswith("refs/tags/v")' in content
    assert '${GITHUB_REF#refs/tags/v}' in content
    assert "startsWith(github.ref, 'refs/tags/v')" in content
    assert broad_tag_release_condition not in content


def test_compileiq_package_release_setup_writes_sourceable_env_file():
    content = MAKEFILE.read_text()

    assert "setup-compileiq-package-release:" in content
    assert "PACKAGE_RELEASE_ENV_FILE ?= dist/compileiq-package-release/current.env" in content
    assert "grep -Eq '^[0-9]+\\.[0-9]+\\.[0-9]+((a|b|rc|dev)[0-9]+)?$$'" in content
    assert 'echo \'export RELEASE_VERSION="$(RELEASE_VERSION)"\'' in content
    assert 'echo \'export RELEASE_TAG="v$(RELEASE_VERSION)"\'' in content
    assert "export RELEASE_MAJOR_MINOR" in content
    assert "export RELEASE_BRANCH" in content
    assert "source $(PACKAGE_RELEASE_ENV_FILE)" in content


def test_search_space_release_prep_is_local_until_publish_path_is_decided():
    content = CI_WORKFLOW.read_text()

    assert "startsWith(github.ref, 'refs/tags/search-spaces-')" not in content


def test_catalog_releases_publish_atomically_without_becoming_latest():
    content = MAKEFILE.read_text()
    release_targets = (
        (
            "publish-search-space-release",
            "clear-search-space-latest",
            "check-search-space-published",
        ),
        (
            "publish-booster-pack-release",
            "clear-booster-pack-latest",
            "check-booster-pack-published",
        ),
    )

    for publish_target, next_target, check_target in release_targets:
        recipe = _make_target_block(content, publish_target, next_target)

        assert "set -eu" in recipe
        assert recipe.count("gh api --method PATCH") == 1
        assert "-F draft=false" in recipe
        assert "-f make_latest=false" in recipe
        assert recipe.index("-F draft=false") < recipe.index("-f make_latest=false")
        assert f"$(MAKE) --no-print-directory {check_target}" in recipe
        assert recipe.index(check_target) < recipe.index('echo "PASS: Published')


def test_ci_workflow_no_longer_deploys_pages_artifacts():
    content = CI_WORKFLOW.read_text()

    assert "deploy-pages:" not in content
    assert "actions/deploy-pages" not in content
    assert "actions/upload-pages-artifact" not in content


def test_docs_workflow_owns_gh_pages_deployment():
    content = DOCS_WORKFLOW.read_text()

    assert "deploy-docs:" in content
    assert "git fetch origin gh-pages:gh-pages" in content
    assert "python dev/deploy_docs.py plan" in content
    assert "python dev/deploy_docs.py deploy" in content
    assert "git push origin gh-pages" in content
    assert "release-[0-9]*.[0-9]*" in content
    assert "v[0-9]*.[0-9]*.[0-9]*" in content
    assert "booster-packs" not in content
    assert "search-spaces" not in content
