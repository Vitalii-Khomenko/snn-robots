#!/usr/bin/env bash
# Renders all robot faces as the robot draws them: docs/faces.png (a grid of 320 x 240 screens) and
# docs/png/FACES_V4.png (2x, nearest neighbour so the pixels stay sharp, every face named: for the website).
# tools/face_preview/run.sh vector  draws the smooth vector faces instead of the pixel ones.
set -Eeuo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"
g++ -std=c++17 -O2 -Wall -Wextra -o "$here/face_preview" "$here/face_preview.cpp" "$root/main/ui/face_style.cpp" \
    "$root/main/ui/pixel_face.cpp"
names="$("$here/face_preview" "$here/faces.ppm" "${1:-pixel}")"
magick "$here/faces.ppm" "$root/docs/faces.png"
echo "wrote $root/docs/faces.png"
# the named sheet: cut the screens apart (21 emotions + 2 remembering), label them, put them together again at 2x
tiles="$here/tiles"
rm -rf "$tiles" && mkdir -p "$tiles"
i=0
for entry in $names; do
  name="${entry#*:}"
  x=$(( (i % 7) * 326 )); y=$(( (i / 7) * 246 ))
  magick "$here/faces.ppm" -crop "320x240+$x+$y" +repage -filter point -resize 200% \
         -background '#06070c' -fill '#8f8ca0' -font "${FACE_FONT:-/usr/share/fonts/noto/NotoSans-Medium.ttf}" -pointsize 26 -gravity south -splice 0x44 \
         -annotate +0+8 "$name" "$tiles/$(printf %02d $i).png"
  i=$((i + 1))
done
mkdir -p "$root/docs/png"
magick montage "$tiles"/*.png -tile 7x -geometry +8+8 -background '#06070c' "$root/docs/png/FACES_V4.png"
rm -rf "$tiles"
echo "wrote $root/docs/png/FACES_V4.png"
