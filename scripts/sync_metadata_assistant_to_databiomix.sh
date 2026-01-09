#!/bin/bash
#
# DEPRECATED: Use sync_to_databiomix.py instead
#
# This wrapper exists for backward compatibility only.
# The new Python script auto-discovers ALL premium files (not just 4).
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Note: This script is deprecated. Using sync_to_databiomix.py instead."
echo ""

exec python3 "$SCRIPT_DIR/sync_to_databiomix.py" "$@"
