#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUTPUT=${1:-$SCRIPT_DIR/bench_cagra}

gcc -O2 -std=c11 -Wall -Wextra \
  -I"$SCRIPT_DIR/.." \
  "$SCRIPT_DIR/bench_cagra.c" \
  -ldl \
  -o "$OUTPUT"

echo "built $OUTPUT"
