#!/usr/bin/env bash
# Installs the PC's user services (the diary receiver, the night sleep, the diary compression) for this clone,
# wherever it is: the unit files name ~/Codex/StackChan (the owner's machine); here they get this clone's path.
#   tools/robot/install_services.sh   then: systemctl --user enable --now lifelogd sleep-v4.timer compress-diary.timer
set -Eeuo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"
dest="$HOME/.config/systemd/user"
mkdir -p "$dest"
for unit in lifelogd.service sleep-v4.service sleep-v4.timer compress-diary.service compress-diary.timer; do
  sed "s|%h/Codex/StackChan|$repo|g" "$here/$unit" > "$dest/$unit"
  echo "installed $dest/$unit"
done
systemctl --user daemon-reload
echo "enable: systemctl --user enable --now lifelogd sleep-v4.timer compress-diary.timer"
