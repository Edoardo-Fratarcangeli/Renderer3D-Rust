//! In-app auto-update client.
//!
//! Checks the GitHub Releases update feed (`latest.json`), verifies the signed
//! artifact against the embedded public key, and installs it in place. The
//! signing key is independent from (and cheaper than) OS code signing — it only
//! protects the update channel from tampering.
//!
//! This stays **inert until configured**: while [`UPDATER_PUBKEY`] is empty the
//! background check returns immediately, so unsigned/dev builds never touch the
//! network. To enable auto-update:
//!   1. Run `scripts/gen-update-key.sh` once.
//!   2. Paste the public key into [`UPDATER_PUBKEY`] below (and into
//!      `Packager.toml`'s `[updater]` block).
//!   3. Add the private key as the `CARGO_PACKAGER_SIGN_PRIVATE_KEY` CI secret.

use cargo_packager_updater::{check_update, Config};

/// Public key used to verify update signatures. Empty = auto-update disabled.
/// Paste the contents of `*.key.pub` from `scripts/gen-update-key.sh` here.
pub const UPDATER_PUBKEY: &str = "";

/// Update manifest endpoint. `latest.json` is published next to the installers
/// on every tagged GitHub Release (see `scripts/gen-latest-json.py`).
pub const UPDATER_ENDPOINT: &str =
    "https://github.com/Edoardo-Fratarcangeli/Renderer3D-Rust/releases/latest/download/latest.json";

/// Whether auto-update has been configured with a public key.
pub fn is_enabled() -> bool {
    !UPDATER_PUBKEY.trim().is_empty()
}

/// Spawn a background thread that checks for, downloads and installs an update.
/// Safe to call unconditionally: it is a no-op until a public key is set, and
/// any network/verification error is logged rather than propagated.
pub fn check_in_background() {
    if !is_enabled() {
        log_info!("Auto-update disabled (no public key configured).");
        return;
    }

    std::thread::spawn(|| {
        if let Err(e) = run_check() {
            crate::log_error!("Auto-update check failed: {e}");
        }
    });
}

fn run_check() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config {
        endpoints: vec![UPDATER_ENDPOINT.parse()?],
        pubkey: UPDATER_PUBKEY.to_string(),
        windows: None,
    };

    let current = env!("CARGO_PKG_VERSION").parse()?;
    match check_update(current, config)? {
        Some(update) => {
            log_info!(
                "Update available: {} (current {}). Downloading…",
                update.version,
                update.current_version
            );
            update.download_and_install()?;
            log_info!("Update installed; it will take effect on next launch.");
        }
        None => {
            log_info!("Application is up to date.");
        }
    }
    Ok(())
}
