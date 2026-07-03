use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Mutex;
use chrono::prelude::*;

// Log Levels
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub enum LogLevel {
    Critical = 0,
    Error = 1,
    Warning = 2,
    Info = 3,
    Verbose = 4,
}

// Global Logger Configuration
static GLOBAL_LOG_LEVEL: AtomicU8 = AtomicU8::new(LogLevel::Verbose as u8);

/// Buffered writer to the day's log file, opened once in [`init`].
/// Buffering avoids an open/write/close syscall trio per line; the buffer is
/// flushed on every Warning/Error/Critical so nothing important is lost on a
/// crash.
static WRITER: Mutex<Option<BufWriter<File>>> = Mutex::new(None);

/// Log directory: `<platform local-data dir>/rendering3d/logs`, mirroring the
/// settings-file convention. A relative "logs/" would depend on the current
/// working directory (and fail outright from a read-only install dir).
fn log_dir() -> PathBuf {
    dirs::data_local_dir()
        .map(|d| d.join("rendering3d").join("logs"))
        .unwrap_or_else(|| PathBuf::from("logs"))
}

pub fn init(level: LogLevel) {
    GLOBAL_LOG_LEVEL.store(level as u8, Ordering::Relaxed);

    let dir = log_dir();
    if let Err(e) = fs::create_dir_all(&dir) {
        eprintln!("Failed to create logs directory '{}': {}", dir.display(), e);
    }
    let path = dir.join(format!("{}_log.txt", Local::now().format("%Y%m%d")));
    match OpenOptions::new().create(true).append(true).open(&path) {
        Ok(file) => {
            // Never propagate poisoning from a panicking thread: logging must
            // stay available for the crash report itself.
            let mut writer = WRITER.lock().unwrap_or_else(|p| p.into_inner());
            *writer = Some(BufWriter::new(file));
        }
        Err(e) => eprintln!("Failed to open log file '{}': {}", path.display(), e),
    }

    log(LogLevel::Info, "Logger initialized.");
}

pub fn log(level: LogLevel, message: &str) {
    if level as u8 > GLOBAL_LOG_LEVEL.load(Ordering::Relaxed) {
        return;
    }

    let now = Local::now();

    // Format: [YYYY-MM-DD HH:MM:SS] [LEVEL] Message
    let level_str = match level {
        LogLevel::Critical => "CRITICAL",
        LogLevel::Error => "ERROR",
        LogLevel::Warning => "WARNING",
        LogLevel::Info => "INFO",
        LogLevel::Verbose => "VERBOSE",
    };

    let log_line = format!("[{}] [{}] {}\n", now.format("%Y-%m-%d %H:%M:%S"), level_str, message);

    // Print to console as well
    print!("{}", log_line);

    let mut writer = WRITER.lock().unwrap_or_else(|p| p.into_inner());
    if let Some(w) = writer.as_mut() {
        if let Err(e) = w.write_all(log_line.as_bytes()) {
            eprintln!("Failed to write log: {}", e);
        }
        // Flush anything at Warning or worse so the tail of the log survives
        // a crash; Info/Verbose ride the buffer.
        if level <= LogLevel::Warning {
            let _ = w.flush();
        }
    }
}

// Macros for convenience
#[macro_export]
macro_rules! log_critical {
    ($($arg:tt)*) => {
        $crate::logger::log($crate::logger::LogLevel::Critical, &format!($($arg)*));
    };
}

#[macro_export]
macro_rules! log_error {
    ($($arg:tt)*) => {
        $crate::logger::log($crate::logger::LogLevel::Error, &format!($($arg)*));
    };
}

#[macro_export]
macro_rules! log_warning {
    ($($arg:tt)*) => {
        $crate::logger::log($crate::logger::LogLevel::Warning, &format!($($arg)*));
    };
}

#[macro_export]
macro_rules! log_info {
    ($($arg:tt)*) => {
        $crate::logger::log($crate::logger::LogLevel::Info, &format!($($arg)*));
    };
}

#[macro_export]
macro_rules! log_verbose {
    ($($arg:tt)*) => {
        $crate::logger::log($crate::logger::LogLevel::Verbose, &format!($($arg)*));
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_levels_are_ordered_from_critical_to_verbose() {
        assert!(LogLevel::Critical < LogLevel::Error);
        assert!(LogLevel::Error < LogLevel::Warning);
        assert!(LogLevel::Warning < LogLevel::Info);
        assert!(LogLevel::Info < LogLevel::Verbose);
        assert_eq!(LogLevel::Critical as u8, 0);
        assert_eq!(LogLevel::Verbose as u8, 4);
    }

    #[test]
    fn log_dir_is_absolute_when_platform_dirs_exist() {
        // On every supported desktop platform data_local_dir() resolves, so
        // the logger must not depend on the process working directory.
        if dirs::data_local_dir().is_some() {
            assert!(log_dir().is_absolute());
        }
    }

    #[test]
    fn init_and_every_level_write_without_panicking() {
        init(LogLevel::Verbose);
        log(LogLevel::Critical, "test critical");
        log(LogLevel::Error, "test error");
        log(LogLevel::Warning, "test warning");
        log(LogLevel::Info, "test info");
        log(LogLevel::Verbose, "test verbose");
    }

    #[test]
    fn messages_above_the_threshold_are_suppressed() {
        // Raising the bar to Critical must drop a Verbose message early; the
        // call still must not panic.
        init(LogLevel::Critical);
        log(LogLevel::Verbose, "should be filtered out");
        // Restore a permissive level for any later logging in the process.
        init(LogLevel::Verbose);
    }

    #[test]
    fn convenience_macros_expand_and_run() {
        init(LogLevel::Verbose);
        log_critical!("c {}", 1);
        log_error!("e {}", 2);
        log_warning!("w {}", 3);
        log_info!("i {}", 4);
        log_verbose!("v {}", 5);
    }
}
