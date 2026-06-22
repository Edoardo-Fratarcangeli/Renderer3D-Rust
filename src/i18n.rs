//! Application interface localization.
//!
//! Translations live in `locales/app.yml` and are embedded at compile time by
//! the `rust_i18n::i18n!` macro in `lib.rs`. This module decides which language
//! to use at startup (saved choice > OS language > English), lets the user
//! switch at runtime, and persists the choice per user.

use std::fs;
use std::path::PathBuf;

/// Languages shipped with the app, in display order: (code, native name).
/// Keep this in sync with the locales available in `locales/app.yml`.
pub const LANGUAGES: &[(&str, &str)] = &[
    ("en", "English"),
    ("it", "Italiano"),
    ("es", "Español"),
    ("fr", "Français"),
    ("de", "Deutsch"),
];

/// Fallback language, also used by `rust_i18n!(fallback = "en")`.
pub const DEFAULT_LANG: &str = "en";

fn is_supported(code: &str) -> bool {
    LANGUAGES.iter().any(|(c, _)| *c == code)
}

fn config_path() -> Option<PathBuf> {
    dirs::config_dir().map(|d| d.join("rendering3d").join("language"))
}

/// Detect the OS UI language (e.g. "it-IT" -> "it"), restricted to the
/// languages we ship; falls back to English.
fn detect_os_language() -> String {
    sys_locale::get_locale()
        .map(|l| l.to_lowercase())
        .and_then(|l| {
            l.split(['-', '_'])
                .next()
                .map(str::to_string)
                .filter(|c| is_supported(c))
        })
        .unwrap_or_else(|| DEFAULT_LANG.to_string())
}

fn load_saved_language() -> Option<String> {
    let code = fs::read_to_string(config_path()?).ok()?.trim().to_string();
    is_supported(&code).then_some(code)
}

/// Resolve and apply the startup language: saved choice > OS language > English.
pub fn init() {
    let lang = load_saved_language().unwrap_or_else(detect_os_language);
    rust_i18n::set_locale(&lang);
}

/// The currently active language code (e.g. "it").
pub fn current() -> String {
    rust_i18n::locale().to_string()
}

/// Switch the UI language at runtime and persist the choice for next launch.
pub fn set_language(code: &str) {
    if !is_supported(code) {
        return;
    }
    rust_i18n::set_locale(code);
    if let Some(path) = config_path() {
        if let Some(dir) = path.parent() {
            let _ = fs::create_dir_all(dir);
        }
        let _ = fs::write(path, code);
    }
}

/// Native display name for a language code, for the language picker.
pub fn display_name(code: &str) -> &'static str {
    LANGUAGES
        .iter()
        .find(|(c, _)| *c == code)
        .map(|(_, name)| *name)
        .unwrap_or("English")
}
