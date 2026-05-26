#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="$ROOT_DIR/public"
DOC_VERSION="${DOC_VERSION:-latest}"
DOC_FOLDER="latest"
if [[ "$DOC_VERSION" != "latest" ]]; then
  DOC_FOLDER="v${DOC_VERSION}"
fi

export CIQ_DOCS_ENV="${CIQ_DOCS_ENV:-dev}"
export CIQ_DOCS_BASE_URL="${CIQ_DOCS_BASE_URL:-http://localhost:8000}"
export CIQ_DOCS_LOCAL_PREVIEW="${CIQ_DOCS_LOCAL_PREVIEW:-1}"
export DOC_VERSION

poetry run sphinx-build -E -a "$ROOT_DIR/docs" "$OUTPUT_DIR/$DOC_FOLDER"
python3 -m http.server 8000 --directory "$OUTPUT_DIR"
