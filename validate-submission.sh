#!/usr/bin/env bash
set -euo pipefail

# Root wrapper for judges/users expecting this script at repository root.
# Delegates to the maintained implementation under scripts/.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/scripts/validate-submission.sh" "$@"

