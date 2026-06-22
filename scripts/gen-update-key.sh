#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  Generates the ed25519 key pair used to SIGN auto-update artifacts.
#  This is independent from (and much cheaper than) Apple / Windows OS code
#  signing — it only protects the auto-update channel from tampering.
#
#  Run this ONCE, locally:
#     cargo install cargo-packager
#     ./scripts/gen-update-key.sh
#
#  Then:
#   1. Put the PRIVATE key in a GitHub Actions secret named
#      CARGO_PACKAGER_SIGN_PRIVATE_KEY  (and its password, if any, in
#      CARGO_PACKAGER_SIGN_PRIVATE_KEY_PASSWORD).
#   2. Paste the PUBLIC key into:
#        - Packager.toml          ([updater] pubkey = "...")
#        - src/updater.rs         (UPDATER_PUBKEY)
#   3. NEVER commit the private key.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

OUT_DIR="${1:-$HOME/.rendering3d-keys}"
mkdir -p "$OUT_DIR"

if ! command -v cargo-packager >/dev/null && ! cargo packager --version >/dev/null 2>&1; then
  echo "cargo-packager not found. Install it with: cargo install cargo-packager" >&2
  exit 1
fi

echo "==> Generating ed25519 update signing key into $OUT_DIR"
cargo packager signer generate -w "$OUT_DIR/rendering3d-update.key"

echo
echo "Private key : $OUT_DIR/rendering3d-update.key      (KEEP SECRET)"
echo "Public  key : $OUT_DIR/rendering3d-update.key.pub  (embed in app + Packager.toml)"
echo
echo "Public key contents:"
cat "$OUT_DIR/rendering3d-update.key.pub"
