# scripts/release.ps1
# Automated Release Script with Semantic Versioning
# Version bump based on git commit messages:
# - "fix:" -> PATCH (x.y.Z)
# - "feat:" -> MINOR (x.Y.0)  
# - "dev:" or "breaking:" -> MAJOR (X.0.0)

param(
    [switch]$DryRun = $false
)

$ErrorActionPreference = "Stop"

# Get project root
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path "$ProjectRoot/Cargo.toml")) {
    $ProjectRoot = (Get-Location).Path
}

Write-Host "Project Root: $ProjectRoot" -ForegroundColor Cyan

# Version file path
$VersionFile = "$ProjectRoot/VERSION"
$ReleasesDir = "$ProjectRoot/releases"

# Read current version or initialize
if (Test-Path $VersionFile) {
    $CurrentVersion = Get-Content $VersionFile -Raw
    $CurrentVersion = $CurrentVersion.Trim()
} else {
    $CurrentVersion = "1.0.0"
    Write-Host "No VERSION file found, starting at $CurrentVersion" -ForegroundColor Yellow
}

Write-Host "Current Version: $CurrentVersion" -ForegroundColor Green

# Parse version
$VersionParts = $CurrentVersion -split '\.'
$Major = [int]$VersionParts[0]
$Minor = [int]$VersionParts[1]
$Patch = [int]$VersionParts[2]

# Analyze git commits since last tag (or all if no tags)
$LastTag = git describe --tags --abbrev=0 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "No previous tags found, analyzing all commits..." -ForegroundColor Yellow
    $Commits = git log --oneline
} else {
    Write-Host "Last Tag: $LastTag" -ForegroundColor Cyan
    $Commits = git log "$LastTag..HEAD" --oneline
}

if (-not $Commits) {
    Write-Host "No new commits since last release. Nothing to do." -ForegroundColor Yellow
    exit 0
}

Write-Host "`nAnalyzing commits..." -ForegroundColor Cyan
$HasDev = $false
$HasFeat = $false
$HasFix = $false

foreach ($commit in $Commits) {
    Write-Host "  $commit" -ForegroundColor Gray
    $lower = $commit.ToLower()
    if ($lower -match "^[a-f0-9]+\s+(dev|breaking):") {
        $HasDev = $true
    }
    elseif ($lower -match "^[a-f0-9]+\s+feat:") {
        $HasFeat = $true
    }
    elseif ($lower -match "^[a-f0-9]+\s+fix:") {
        $HasFix = $true
    }
}

# Determine version bump
if ($HasDev) {
    $Major++
    $Minor = 0
    $Patch = 0
    Write-Host "`nMAJOR bump detected (dev/breaking commit)" -ForegroundColor Magenta
}
elseif ($HasFeat) {
    $Minor++
    $Patch = 0
    Write-Host "`nMINOR bump detected (feat commit)" -ForegroundColor Yellow
}
elseif ($HasFix) {
    $Patch++
    Write-Host "`nPATCH bump detected (fix commit)" -ForegroundColor Green
}
else {
    $Patch++
    Write-Host "`nNo conventional commits found, defaulting to PATCH bump" -ForegroundColor Gray
}

$NewVersion = "$Major.$Minor.$Patch"
Write-Host "`nNew Version: $NewVersion" -ForegroundColor Cyan

if ($DryRun) {
    Write-Host "`n[DRY RUN] Would release version $NewVersion" -ForegroundColor Yellow
    exit 0
}

# Build release
Write-Host "`nBuilding release..." -ForegroundColor Cyan
Set-Location $ProjectRoot
cargo build --release
if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    exit 1
}

# Create release directory
$ReleaseDir = "$ReleasesDir/$NewVersion"
if (Test-Path $ReleaseDir) {
    Write-Host "Release directory already exists: $ReleaseDir" -ForegroundColor Yellow
    Write-Host "Overwriting..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $ReleaseDir
}

New-Item -ItemType Directory -Path $ReleaseDir -Force | Out-Null

# Copy executable
$ExePath = "$ProjectRoot/target/release/rendering_3d.exe"
if (Test-Path $ExePath) {
    Copy-Item $ExePath "$ReleaseDir/"
    Write-Host "Copied: rendering_3d.exe" -ForegroundColor Green
} else {
    Write-Host "Executable not found: $ExePath" -ForegroundColor Red
    exit 1
}

# Copy any required assets (if any exist)
$AssetsDir = "$ProjectRoot/assets"
if (Test-Path $AssetsDir) {
    Copy-Item -Recurse $AssetsDir "$ReleaseDir/"
    Write-Host "Copied: assets/" -ForegroundColor Green
}

# Create version info file
@"
Version: $NewVersion
Built: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
Commit: $(git rev-parse --short HEAD)
"@ | Out-File "$ReleaseDir/VERSION.txt" -Encoding utf8

# Update VERSION file
$NewVersion | Out-File $VersionFile -Encoding utf8 -NoNewline

# Create git tag
Write-Host "`nCreating git tag v$NewVersion..." -ForegroundColor Cyan
git tag -a "v$NewVersion" -m "Release v$NewVersion"

Write-Host "`n========================================" -ForegroundColor Green
Write-Host " Release $NewVersion created successfully!" -ForegroundColor Green
Write-Host " Location: $ReleaseDir" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "`nTo push the tag: git push origin v$NewVersion" -ForegroundColor Yellow
