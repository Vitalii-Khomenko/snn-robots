#!/usr/bin/env bash
# Redraws the brain maps from the code (tools/v4_map.py), renders every docs/*.svg to a high-quality PNG for the website
# (docs/png/<name>.png, 2x: 4800 px wide, the SVG's own dark background) and pushes the maps to GitHub.
#
#   tools/publish_maps.sh            redraw, render, commit and push (only docs/*.svg and docs/png/ are committed)
#   tools/publish_maps.sh --no-push  redraw and render only
set -Eeuo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "$here/.." && pwd)"
command -v rsvg-convert >/dev/null || { echo "rsvg-convert is missing (pacman -S librsvg)" >&2; exit 1; }

"$here/v4/py" ../v4_map.py
mkdir -p "$project/docs/png"
for svg in "$project"/docs/*.svg; do
  png="$project/docs/png/$(basename "${svg%.svg}").png"
  rsvg-convert --zoom 2 --keep-aspect-ratio "$svg" -o "$png"
  echo "rendered $png ($(du -h "$png" | cut -f1))"
done

[[ "${1:-}" == "--no-push" ]] && exit 0
cd "$project"
paths=(docs/*.svg docs/png)
git add -- "${paths[@]}"
if git diff --cached --quiet -- "${paths[@]}"; then
  echo "maps unchanged, nothing to push"
  exit 0
fi
git commit -q -m "Brain v4 maps: redrawn from the code, PNG renders for the website" \
  -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- "${paths[@]}"
git push -q origin HEAD
echo "pushed: $(git log --oneline -1)"
