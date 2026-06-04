#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'USAGE'
usage: dev/update-core-binaries.sh --manifest-url URL_OR_PATH --tarball-url URL_OR_PATH [--expected-manifest-sha256 SHA256]

Fetches a core manifest and tarball, verifies the extracted tree against the
manifest, then replaces compileiq/core/executable with the verified files.

Inputs may be HTTPS URLs, file:// URLs, or local filesystem paths.
USAGE
}

MANIFEST_URL=""
TARBALL_URL=""
EXPECTED_MANIFEST_SHA256=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --manifest-url)
            MANIFEST_URL="${2:-}"
            shift 2
            ;;
        --tarball-url)
            TARBALL_URL="${2:-}"
            shift 2
            ;;
        --expected-manifest-sha256)
            EXPECTED_MANIFEST_SHA256="${2:-}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage
            exit 2
            ;;
    esac
done

if [[ -z "${MANIFEST_URL}" || -z "${TARBALL_URL}" ]]; then
    usage
    exit 2
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXEC_DIR="${REPO}/compileiq/core/executable"
TMP_BASE="${TMPDIR:-/tmp}"
WORK="$(mktemp -d "${TMP_BASE%/}/ciq-core.XXXXXX")"
trap 'rm -rf "${WORK}"' EXIT

download() {
    local url="$1"
    local out="$2"

    if [[ -f "${url}" ]]; then
        cp "${url}" "${out}"
        return
    fi

    if [[ "${url}" == file://* ]]; then
        local path="${url#file://}"
        if [[ ! -f "${path}" ]]; then
            echo "local artifact does not exist: ${path}" >&2
            exit 1
        fi
        cp "${path}" "${out}"
        return
    fi

    curl -fL --retry 3 --retry-delay 2 --show-error --silent -o "${out}" "${url}"
}

sha256() {
    python3 -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$1"
}

echo ">>> Downloading core manifest"
download "${MANIFEST_URL}" "${WORK}/core-manifest.json"

MANIFEST_SHA256="$(sha256 "${WORK}/core-manifest.json")"
if [[ -n "${EXPECTED_MANIFEST_SHA256}" ]]; then
    EXPECTED_MANIFEST_SHA256="${EXPECTED_MANIFEST_SHA256#sha256:}"
    if [[ "${MANIFEST_SHA256}" != "${EXPECTED_MANIFEST_SHA256}" ]]; then
        echo "manifest hash mismatch" >&2
        echo "  expected sha256:${EXPECTED_MANIFEST_SHA256}" >&2
        echo "  actual   sha256:${MANIFEST_SHA256}" >&2
        exit 1
    fi
fi

echo ">>> Downloading core tarball"
download "${TARBALL_URL}" "${WORK}/core-binaries.tar.gz"

mkdir -p "${WORK}/extract"
tar -xzf "${WORK}/core-binaries.tar.gz" -C "${WORK}/extract"

python3 "${REPO}/dev/public_core_manifest.py" \
    --source "${WORK}/core-manifest.json" \
    --output "${WORK}/core-manifest.public.json" \
    --source-manifest-sha256 "${MANIFEST_SHA256}"

echo ">>> Verifying extracted core files"
(cd "${REPO}" && python3 -m compileiq.core.verify_core \
    --executable-root "${WORK}/extract" \
    --manifest "${WORK}/core-manifest.public.json")

echo ">>> Replacing ${EXEC_DIR}"
OLD_EXEC_DIR="${EXEC_DIR}.old.$$"
mv "${EXEC_DIR}" "${OLD_EXEC_DIR}"
mv "${WORK}/extract" "${EXEC_DIR}"
cp "${WORK}/core-manifest.public.json" "${EXEC_DIR}/core-manifest.json"

echo ">>> Verifying installed core files"
(cd "${REPO}" && python3 -m compileiq.core.verify_core)
rm -rf "${OLD_EXEC_DIR}"
