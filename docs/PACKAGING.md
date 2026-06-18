# Packaging & Installers

This document describes how Rust 3D Renderer is turned into **self-contained,
native installers** for Windows, macOS and Linux, how releases are automated,
and how the multilingual installers and in-app auto-update work.

## Design at a glance

The binary is already self-contained: the shader and the window icon are embedded
with `include_bytes!`, so the app needs **no external files at runtime**. Packaging
therefore only wraps that single binary into each platform's native installer.

Because a `.dmg`/`.app` cannot be built on Linux nor an `.exe` installer reliably
on macOS, the architecture is a **CI matrix with one native runner per OS**:

| Platform | Installer(s) | Tool | Multilingual UI |
|----------|--------------|------|-----------------|
| Windows  | `*-setup.exe` (NSIS, MUI2) | `makensis` (custom script) | language selector + EN/IT/ES/FR/DE pages |
| macOS    | `.dmg` (drag-to-Applications) + `.pkg` (wizard) | `cargo-packager` + `productbuild` | localized welcome/readme/conclusion panes |
| Linux    | `.AppImage` (portable) + `.deb` | `cargo-packager` | localized `.desktop` + AppStream metadata |

- **AppImage** is the most self-contained Linux format (bundles the Vulkan/X11/
  Wayland client libraries the GPU stack needs). The `.deb` declares those as
  dependencies instead.
- **macOS** ships a universal (arm64 + x86_64) binary.

## Files

```
Packager.toml                         cargo-packager config (macOS + Linux)
.github/workflows/release.yml         CI matrix: build + package + GitHub Release
scripts/gen-icons.sh                  assets/icon.svg -> .png/.ico/.icns set
scripts/gen-update-key.sh             one-time ed25519 update-signing key
scripts/gen-latest-json.py            builds the auto-update manifest
packaging/windows/installer.nsi       custom multilingual NSIS wizard
packaging/windows/strings/*.nsh       EN/IT/ES/FR/DE installer strings
packaging/macos/build-pkg.sh          builds the localized .pkg wizard
packaging/macos/distribution.xml      .pkg wizard layout
packaging/macos/resources/<lang>.lproj localized welcome/readme/conclusion panes
packaging/linux/rendering-3d.desktop  localized desktop entry
packaging/linux/*.metainfo.xml        localized AppStream metadata (software centers)
```

## Releasing (automated)

1. Bump the version (`VERSION`, `Cargo.toml`) and commit. Keep `Cargo.toml`'s
   `version` in sync with the tag: the in-app updater compares the running
   `CARGO_PKG_VERSION` against the release, so a stale value breaks update
   detection.
2. Tag and push:
   ```bash
   git tag v1.2.0
   git push origin v1.2.0
   ```
3. `release.yml` builds all installers on native runners and publishes a GitHub
   Release with the artifacts attached. (You can also trigger it manually from the
   Actions tab via *workflow_dispatch*.)

The existing `scripts/release.ps1` can still be used upstream to compute the
semver bump from commit messages and create the tag.

## Building locally

```bash
# Linux / macOS
cargo install cargo-packager --locked
./scripts/gen-icons.sh
cargo build --release
cargo packager --release            # reads Packager.toml

# Windows (custom NSIS wizard)
choco install nsis imagemagick -y
bash scripts/gen-icons.sh
cargo build --release
makensis -DAPP_VERSION=1.1.0 packaging/windows/installer.nsi
```

## Multilingual installers

- **Windows**: the first page is a language selector (`MUI_LANGDLL_DISPLAY`).
  All wizard text — welcome, the "What is this software?" page, components and
  finish — comes from `packaging/windows/strings/<Language>.nsh`. Add a language
  by adding a `MUI_LANGUAGE` line in `installer.nsi` plus a matching `.nsh`.
- **macOS**: `productbuild` resolves the welcome/readme/conclusion panes from
  `packaging/macos/resources/<lang>.lproj/`. Add a language by adding a `.lproj`
  folder.
- **Linux**: localized `Name`/`Comment` in the `.desktop` file and localized
  `summary`/`description` in the AppStream `metainfo.xml` are shown by GNOME
  Software / KDE Discover in the user's language.

The **application's own UI language** is independent and lives in `locales/`
(see [I18N.md](I18N.md)); it auto-detects the OS language and can be changed from
in-app Settings.

## Code signing (deferred)

The installers are currently **unsigned**, so users see a one-time security
prompt on first launch (macOS Gatekeeper, Windows SmartScreen). To remove those:

- **macOS**: an Apple Developer ID (~99 €/yr). Set `signing-identity` in
  `[macos]` and add notarization (`xcrun notarytool`) in CI. *Recommended first*,
  because an unsigned auto-updated `.app` can fail Gatekeeper re-validation.
- **Windows**: an OV/EV code-signing certificate; sign `*-setup.exe` with
  `signtool` in CI.

Neither is required to ship; they only remove the warnings.

## Auto-update

Auto-update is wired but **off until you create the signing key** (and it needs a
small amount of app code — see `src/updater.rs`). Steps:

1. Generate the ed25519 key **once**:
   ```bash
   ./scripts/gen-update-key.sh
   ```
2. Add GitHub Actions secrets:
   - `CARGO_PACKAGER_SIGN_PRIVATE_KEY` — the private key contents
   - `CARGO_PACKAGER_SIGN_PRIVATE_KEY_PASSWORD` — its password (if set)
3. Paste the **public** key into `Packager.toml` (`[updater]`, uncomment it) and
   into `src/updater.rs` (`UPDATER_PUBKEY`).
4. On the next tagged release, CI signs every artifact (`*.sig`) and
   `gen-latest-json.py` publishes `latest.json` alongside them. The app polls that
   URL, verifies the signature against the embedded public key, and updates in place.

> The update-signing key is **separate from and cheaper than** OS code signing.
> It only protects the update channel from tampering; it does not silence
> Gatekeeper/SmartScreen.

## Troubleshooting

- **Linux app won't start (Vulkan)** — install the loader (`libvulkan1`) or use the
  AppImage, which bundles it.
- **`makensis` not found on Windows** — it installs to
  `C:\Program Files (x86)\NSIS`; the workflow adds that to `PATH`.
- **`.icns` not generated** — install `libicns` (`png2icns`) or build on macOS where
  `iconutil` is available; the workflow installs `icnsutils` on Linux.
