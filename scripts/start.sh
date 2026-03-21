#!/usr/bin/env bash
# Start the AIO Gateway (FastAPI + Telegram bot)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Load environment variables from config/.env
ENV_FILE="${PROJECT_DIR}/config/.env"
if [ -f "$ENV_FILE" ]; then
    set -a
    # shellcheck source=/dev/null
    source "$ENV_FILE"
    set +a
    echo "Loaded env from $ENV_FILE"
else
    echo "Warning: $ENV_FILE not found, using existing environment"
fi

cd "$PROJECT_DIR"

exec uv run aio serve "$@"
