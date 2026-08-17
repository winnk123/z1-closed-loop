#!/usr/bin/env bash
set -eo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/load_z1_env.sh"
exec python3 "$script_dir/check_z1_closed_loop_readiness.py" "$@"
