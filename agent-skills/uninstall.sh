#!/usr/bin/env bash
# Convenience wrapper: uninstall.sh == install.sh --uninstall.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/install.sh" --uninstall "$@"
