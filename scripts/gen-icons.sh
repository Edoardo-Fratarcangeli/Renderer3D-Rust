#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  Generates the cross-platform icon set from assets/icon.svg into
#  packaging/icons/:
#     32x32.png  128x128.png  128x128@2x.png  icon.png (512)   -> Linux/cargo-packager
#     icon.ico                                                  -> Windows
#     icon.icns                                                 -> macOS
#
#  Dependencies (any one rasterizer + platform packers):
#     rasterizer : rsvg-convert | inkscape | magick/convert
#     .ico       : magick/convert  (ImageMagick)
#     .icns      : iconutil (macOS)  OR  png2icns (libicns, Linux/CI)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SVG="$ROOT/assets/icon.svg"
OUT="$ROOT/packaging/icons"
mkdir -p "$OUT"

render() { # render <size> <outfile>
  local size="$1" out="$2"
  if command -v rsvg-convert >/dev/null; then
    rsvg-convert -w "$size" -h "$size" "$SVG" -o "$out"
  elif command -v inkscape >/dev/null; then
    inkscape "$SVG" --export-type=png --export-filename="$out" -w "$size" -h "$size" >/dev/null 2>&1
  elif command -v magick >/dev/null; then
    magick -background none -density 384 "$SVG" -resize "${size}x${size}" "$out"
  elif command -v convert >/dev/null; then
    convert -background none -density 384 "$SVG" -resize "${size}x${size}" "$out"
  else
    echo "ERROR: need rsvg-convert, inkscape or ImageMagick to rasterize the SVG" >&2
    exit 1
  fi
}

echo "==> Rasterizing PNGs"
render 32  "$OUT/32x32.png"
render 128 "$OUT/128x128.png"
render 256 "$OUT/128x128@2x.png"
render 512 "$OUT/icon.png"
render 16  "$OUT/16.png"
render 48  "$OUT/48.png"
render 1024 "$OUT/1024.png"

echo "==> Building .ico (Windows)"
if command -v magick >/dev/null; then
  magick "$OUT/16.png" "$OUT/32x32.png" "$OUT/48.png" "$OUT/128x128.png" "$OUT/128x128@2x.png" "$OUT/icon.ico"
elif command -v convert >/dev/null; then
  convert "$OUT/16.png" "$OUT/32x32.png" "$OUT/48.png" "$OUT/128x128.png" "$OUT/icon.ico"
else
  echo "   (ImageMagick missing) reusing existing assets/icon.ico"
  cp "$ROOT/assets/icon.ico" "$OUT/icon.ico"
fi

echo "==> Building .icns (macOS)"
if command -v iconutil >/dev/null; then
  ICONSET="$(mktemp -d)/icon.iconset"; mkdir -p "$ICONSET"
  render 16   "$ICONSET/icon_16x16.png"
  render 32   "$ICONSET/icon_16x16@2x.png"
  render 32   "$ICONSET/icon_32x32.png"
  render 64   "$ICONSET/icon_32x32@2x.png"
  render 128  "$ICONSET/icon_128x128.png"
  render 256  "$ICONSET/icon_128x128@2x.png"
  render 256  "$ICONSET/icon_256x256.png"
  render 512  "$ICONSET/icon_256x256@2x.png"
  render 512  "$ICONSET/icon_512x512.png"
  render 1024 "$ICONSET/icon_512x512@2x.png"
  iconutil -c icns "$ICONSET" -o "$OUT/icon.icns"
elif command -v png2icns >/dev/null; then
  png2icns "$OUT/icon.icns" "$OUT/16.png" "$OUT/32x32.png" "$OUT/48.png" "$OUT/128x128.png" "$OUT/128x128@2x.png" "$OUT/icon.png" || true
else
  echo "   (no iconutil/png2icns) .icns not generated — install libicns (png2icns) in CI"
fi

rm -f "$OUT/16.png" "$OUT/48.png" "$OUT/256.png" "$OUT/1024.png" 2>/dev/null || true
echo "==> Icon set ready in $OUT"
ls -1 "$OUT"
