use std::fs::{self, OpenOptions};
use std::io::Write;
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
static mut GLOBAL_LOG_LEVEL: LogLevel = LogLevel::Verbose;

// Simple file path storage isn't strictly needed if we just open append every time 
// or keep a static file handle (harder with safe Rust without lazy_static/OnceLock).
// For simplicity and robustness, we'll open-append on each log for now, or use a Mutex<Option<File>>.

static FILE_MUTEX: Mutex<()> = Mutex::new(());

pub fn init(level: LogLevel) {
    unsafe {
        GLOBAL_LOG_LEVEL = level;
    }
    
    // Create logs directory
    if let Err(e) = fs::create_dir_all("logs") {
        eprintln!("Failed to create logs directory: {}", e);
    }
    
    log(LogLevel::Info, "Logger initialized.");
}

pub fn log(level: LogLevel, message: &str) {
    unsafe {
        if level > GLOBAL_LOG_LEVEL {
            return;
        }
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
