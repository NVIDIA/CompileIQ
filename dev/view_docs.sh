#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="$ROOT_DIR/public"

export CIQ_DOCS_ENV="${CIQ_DOCS_ENV:-dev}"
export CIQ_DOCS_BASE_URL="${CIQ_DOCS_BASE_URL:-http://localhost:8000}"

poetry run sphinx-multiversion -E -a "$ROOT_DIR/docs" "$OUTPUT_DIR"
python3 -m http.server 8000 --directory "$OUTPUT_DIR"