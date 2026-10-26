#!/usr/bin/env bash
# Regression test for https://github.com/oceanbase/seekdb/issues/1544
#
# Start a fresh seekdb instance with a deliberately tiny RPC timeout. Bootstrap
# must still use the internal SQL timeout budget for init_global_stat rather
# than allowing the short RPC budget to abort the first start.
#
# Usage: tools/ci/bootstrap_timeout_smoke_test.sh <seekdb-binary> [timeout-seconds]

set -euo pipefail

seekdb_bin="${1:-}"
timeout_seconds="${2:-180}"
if [[ -z "${seekdb_bin}" || ! -x "${seekdb_bin}" ]]; then
  echo "usage: $0 <seekdb-binary> [timeout-seconds]" >&2
  exit 2
fi

base_dir="$(mktemp -d "${TMPDIR:-/tmp}/seekdb-bootstrap-timeout.XXXXXX")"
console_log="${base_dir}/console.log"
pid=""

cleanup() {
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    kill "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
  fi
  rm -rf "${base_dir}"
}
trap cleanup EXIT

port="$(
  python3 -c '
import socket
sock = socket.socket()
sock.bind(("127.0.0.1", 0))
print(sock.getsockname()[1])
sock.close()
'
)"

echo "[bootstrap-timeout] starting ${seekdb_bin} with rpc_timeout=1ms"
"${seekdb_bin}" \
  --nodaemon \
  --base-dir="${base_dir}" \
  --port="${port}" \
  --parameter rpc_timeout=1ms \
  >"${console_log}" 2>&1 &
pid=$!

deadline=$((SECONDS + timeout_seconds))
while (( SECONDS < deadline )); do
  if ! kill -0 "${pid}" 2>/dev/null; then
    status=0
    wait "${pid}" || status=$?
    echo "[bootstrap-timeout] FAIL: seekdb exited before readiness (status ${status})" >&2
    tail -n 160 "${console_log}" >&2 || true
    find "${base_dir}" -type f -name '*.log' -print -exec tail -n 160 {} \; >&2 || true
    exit 1
  fi
  if grep -Rqs 'observer instance start success' \
      "${console_log}" "${base_dir}/log" 2>/dev/null; then
    echo "[bootstrap-timeout] PASS: fresh database bootstrapped and became ready"
    exit 0
  fi
  sleep 1
done

echo "[bootstrap-timeout] FAIL: readiness timed out after ${timeout_seconds}s" >&2
tail -n 160 "${console_log}" >&2 || true
find "${base_dir}" -type f -name '*.log' -print -exec tail -n 160 {} \; >&2 || true
exit 1
