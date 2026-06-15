use std::fs::{self, OpenOptions};
use std::io::Write;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU8, Ordering};
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

// Simple file path storage isn't strictly needed if we just open append every time 
// or keep a static file handle (harder with safe Rust without lazy_static/OnceLock).
// For simplicity and robustness, we'll open-append on each log for now, or use a Mutex<Option<File>>.

static FILE_MUTEX: Mutex<()> = Mutex::new(());

pub fn init(level: LogLevel) {
    GLOBAL_LOG_LEVEL.store(level as u8, Ordering::Relaxed);
    
    // Create logs directory
    if let Err(e) = fs::create_dir_all("logs") {
        eprintln!("Failed to create logs directory: {}", e);
    }
    
    log(LogLevel::Info, "Logger initialized.");
}

pub fn log(level: LogLevel, message: &str) {
    if level as u8 > GLOBAL_LOG_LEVEL.load(Ordering::Relaxed) {
        return;
    }

    let now = Local::now();
    let filename = format!("logs/{}_log.txt", now.format("%Y%m%d"));
    
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

    // Write to file
    let _lock = FILE_MUTEX.lock().unwrap(); // Simple lock to prevent interleaved writes
    match OpenOptions::new().create(true).append(true).open(&filename) {
        Ok(mut file) => {
            if let Err(e) = file.write_all(log_line.as_bytes()) {
                eprintln!("Failed to write log: {}", e);
            }
        }
        Err(e) => {
            eprintln!("Failed to open log file '{}': {}", filename, e);
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
