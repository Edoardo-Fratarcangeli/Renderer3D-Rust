# scripts/build-installer.ps1
# Builds the Windows NSIS installer locally, replicating the CI pipeline.
#
# Hard requirements (script exits if missing):
#   - NSIS  (makensis) -- choco install nsis  or  https://nsis.sourceforge.io/Download
#   - Rust  (cargo)    -- https://rustup.rs
#
# Optional:
#   - Git Bash (bash)      -- needed to regenerate packaging/icons/ from assets/icon.svg
#   - ImageMagick (magick) -- needed to regenerate header.bmp / sidebar.bmp
#     If either optional tool is absent the step is skipped; the installer still
#     builds because the .nsi script has !if /FileExists guards for all artwork.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\build-installer.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\build-installer.ps1 -Version 1.2.0

param([string]$Version = "")

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot

if (-not $Version) {
    $Version = (Get-Content "$Root\VERSION" -Raw).Trim()
}

# Augment PATH with common local install locations for NSIS and ImageMagick.
$extraPaths = @("C:\Program Files (x86)\NSIS", "C:\Program Files\ImageMagick")
# Pick up any versioned ImageMagick install (e.g. ImageMagick-7.1.2-Q16-HDRI).
$extraPaths += Get-ChildItem "C:\Program Files" -Filter "ImageMagick-*" -Directory -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty FullName
foreach ($candidate in $extraPaths) {
    if ((Test-Path $candidate) -and $env:PATH -notlike "*$candidate*") {
        $env:PATH = "$candidate;$env:PATH"
    }
}

Write-Host "==> Building Windows installer v$Version" -ForegroundColor Cyan
Write-Host "    Root : $Root"

# Hard requirements.
foreach ($tool in @("makensis", "cargo")) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        Write-Host "ERROR: '$tool' not found in PATH." -ForegroundColor Red
        if ($tool -eq "makensis") { Write-Host "  -> choco install nsis  or  https://nsis.sourceforge.io/Download" }
        if ($tool -eq "cargo")    { Write-Host "  -> https://rustup.rs" }
        exit 1
    }
}

# Prefer Git Bash over WSL bash (WSL bash needs a Linux distro installed).
$gitBash = @(
    "C:\Program Files\Git\bin\bash.exe",
    "C:\Program Files\Git\usr\bin\bash.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1
$hasBash   = [bool]$gitBash
$hasMagick = [bool](Get-Command magick -ErrorAction SilentlyContinue)

$icons  = "$Root\packaging\icons"
$winPkg = "$Root\packaging\windows"

# --- 1. Icon set (optional - needs bash + gen-icons.sh) ---
if ($hasBash) {
    Write-Host "`n[1/4] Generating icon set from assets\icon.svg ..." -ForegroundColor Cyan
    $rootUnix = ($Root -replace '\\', '/')
    & $gitBash "$rootUnix/scripts/gen-icons.sh"
} else {
    Write-Host "`n[1/4] Git Bash not found -- skipping icon regeneration." -ForegroundColor Yellow
    Write-Host "       (Install Git for Windows to enable this step)"
}

# --- 2. Installer artwork (optional - needs ImageMagick) ---
if ($hasMagick) {
    Write-Host "`n[2/4] Generating installer artwork (header.bmp, sidebar.bmp) ..." -ForegroundColor Cyan

    # header.bmp -- 150x57 dark banner with icon left and app name right
    & magick @(
        "-size", "150x57", "xc:#1b1f24",
        "(", "$icons\128x128.png", "-resize", "44x44", ")",
        "-gravity", "west", "-geometry", "+8+0", "-composite",
        "-fill", "white", "-gravity", "east", "-pointsize", "13",
        "-annotate", "+10+0", "Rust 3D Renderer",
        "BMP3:$winPkg\header.bmp"
    )

    # sidebar.bmp -- 164x314 dark gradient with large icon centered top
    & magick @(
        "-size", "164x314", "gradient:#2b3340-#11151b",
        "(", "$icons\icon.png", "-resize", "110x110", ")",
        "-gravity", "north", "-geometry", "+0+40", "-composite",
        "BMP3:$winPkg\sidebar.bmp"
    )
} else {
    $headerExists  = Test-Path "$winPkg\header.bmp"
    $sidebarExists = Test-Path "$winPkg\sidebar.bmp"
    if ($headerExists -and $sidebarExists) {
        Write-Host "`n[2/4] ImageMagick not found -- using existing BMP files." -ForegroundColor Yellow
    } else {
        Write-Host "`n[2/4] ImageMagick not found -- artwork will use NSIS defaults." -ForegroundColor Yellow
        Write-Host "       (Install with: choco install imagemagick)"
    }
}

# --- 3. Rust release binary ---
Write-Host "`n[3/4] Building Rust release binary ..." -ForegroundColor Cyan
Push-Location $Root
try {
    cargo build --release
    if ($LASTEXITCODE -ne 0) { throw "cargo build failed" }
} finally {
    Pop-Location
}

# --- 4. NSIS ---
# makensis resolves !include relative paths from the cwd, so run from project root.
Write-Host "`n[4/4] Running makensis -DAPP_VERSION=$Version ..." -ForegroundColor Cyan
New-Item -ItemType Directory -Force "$Root\dist" | Out-Null
Push-Location $Root
try {
    & makensis `
        "-DAPP_VERSION=$Version" `
        "-DSRC_DIR=$Root\target\release" `
        "-DOUT_FILE=$Root\dist\Rust-3D-Renderer-$Version-x64-setup.exe" `
        "packaging\windows\installer.nsi"
    if ($LASTEXITCODE -ne 0) { throw "makensis failed" }
} finally {
    Pop-Location
}

$Exe = "$Root\dist\Rust-3D-Renderer-$Version-x64-setup.exe"
if (Test-Path $Exe) {
    Write-Host "`n OK  Installer ready: $Exe" -ForegroundColor Green
} else {
    Write-Host "`n ERR Installer not found at expected path: $Exe" -ForegroundColor Red
    exit 1
}
