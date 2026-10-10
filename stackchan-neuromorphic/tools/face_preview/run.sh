#!/usr/bin/env bash
# Renders all robot faces into docs/faces.png (and the scratch PPM).
set -Eeuo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"
g++ -std=c++17 -O2 -Wall -Wextra -o "$here/face_preview" "$here/face_preview.cpp" "$root/main/ui/face_style.cpp"
"$here/face_preview" "$here/faces.ppm"
out="${1:-$root/docs/faces.png}"
convert "$here/faces.ppm" -gravity north -splice 0x0 "$out"
echo "wrote $out"
