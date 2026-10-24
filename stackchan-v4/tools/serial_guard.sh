#!/usr/bin/env bash
# Source after setting port. FD 9 holds the lock for the entire operation.
port="$(readlink -f "$port")"
if [[ ! -c "$port" ]]; then
  echo "ERROR: serial device missing: $port" >&2
  exit 2
fi
exec 9>"/tmp/stackchan-$(basename "$port").lock"
flock -n 9 || { echo "ERROR: another project serial operation owns $port" >&2; exit 2; }
if fuser "$port" >/dev/null 2>&1; then
  echo "ERROR: $port is open; close its monitor first." >&2
  fuser -v "$port" >&2 || true
  exit 2
fi
# Root-owned orphan Docker monitors may be invisible to unprivileged fuser.
# Check actual device mappings, not container names.
serial_containers="$(docker ps -q)"
if [[ -n "$serial_containers" ]]; then
  while read -r serial_container; do
    if docker inspect --format '{{range .HostConfig.Devices}}{{println .PathOnHost}}{{end}}' "$serial_container" |
         grep -Fxq "$port"; then
      echo "ERROR: container $serial_container maps $port; stop that serial consumer first." >&2
      exit 2
    fi
  done <<< "$serial_containers"
fi
