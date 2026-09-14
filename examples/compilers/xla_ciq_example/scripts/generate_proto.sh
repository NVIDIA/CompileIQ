#!/usr/bin/env bash
# Regenerate Python bindings from schemas/xla_ciq_bundle.proto.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
python -m grpc_tools.protoc \
  -I "$ROOT/schemas" \
  --python_out="$ROOT/src/xla_ciq" \
  "$ROOT/schemas/xla_ciq_bundle.proto"
echo "Wrote $ROOT/src/xla_ciq/xla_ciq_bundle_pb2.py"
