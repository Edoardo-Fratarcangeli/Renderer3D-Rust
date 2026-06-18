#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  Builds a localized macOS installer wizard (.pkg) around the .app bundle that
#  cargo-packager produced. The .pkg shows welcome / read-me / conclusion panes
#  localized into English, Italian, Spanish, French and German, and installs the
#  app into /Applications.
#
#  Usage:
#    packaging/macos/build-pkg.sh <path/to/App.app> <version> <output.pkg>
#
#  Notes:
#   - Runs only on macOS (uses pkgbuild / productbuild).
#   - The .dmg (drag-to-Applications) is still produced by cargo-packager; this
#     .pkg is the explanatory, wizard-style alternative.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

APP_PATH="${1:?usage: build-pkg.sh <App.app> <version> <output.pkg>}"
VERSION="${2:?missing version}"
OUT="${3:?missing output path}"

IDENTIFIER="tech.fratarcangeli.rendering3d"
ROOT="$(cd "$(dirname "$0")" && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "==> Building component package from $APP_PATH"
pkgbuild \
  --component "$APP_PATH" \
  --install-location "/Applications" \
  --identifier "$IDENTIFIER" \
  --version "$VERSION" \
  "$WORK/core.pkg"

echo "==> Building localized product archive -> $OUT"
mkdir -p "$(dirname "$OUT")"
productbuild \
  --distribution "$ROOT/distribution.xml" \
  --resources "$ROOT/resources" \
  --package-path "$WORK" \
  "$OUT"

echo "==> Done: $OUT"
