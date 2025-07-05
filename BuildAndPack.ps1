# BuildAndPack.ps1 - Simple version increment, build and pack for Firefly

param(
    [Switch]$SkipBuild,
    [String]$Configuration = "Release"
)

$ErrorActionPreference = "Stop"

# Paths
$SrcPath = Join-Path $PSScriptRoot "src"
$ProjectFile = Join-Path $SrcPath "Firefly.fsproj"

function Write-Status {
    param([string]$Message)
    Write-Host "-> $Message" -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host "✓ $Message" -ForegroundColor Green
}

function Write-Fail {
    param([string]$Message)
    Write-Host "✗ $Message" -ForegroundColor Red
}

# Check prerequisites
Write-Status "Checking prerequisites..."
if (-not (Test-Path $ProjectFile)) {
    Write-Fail "Project file not found: $ProjectFile"
    exit 1
}

try {
    dotnet --version | Out-Null
    Write-Success "dotnet CLI available"
}
catch {
    Write-Fail "dotnet CLI not found"
    exit 1
}

# Version increment (only if building)
if (-not $SkipBuild) {
    Write-Status "Reading current version..."

    $content = Get-Content $ProjectFile -Raw
    $versionMatch = $content -match '<Version>(\d+)\.(\d+)\.(\d+)</Version>'

    if (-not $versionMatch) {
        Write-Fail "Could not find version in format <Version>0.1.004</Version>"
        exit 1
    }

    $major = $matches[1]
    $minor = $matches[2]
    $build = [int]$matches[3] + 1
    $newVersion = "$major.$minor.$($build.ToString('000'))"

    Write-Status "Updating version to $newVersion..."
    $newContent = $content -replace '<Version>\d+\.\d+\.\d+</Version>', "<Version>$newVersion</Version>"
    Set-Content $ProjectFile $newContent

    Write-Success "Version updated to $newVersion"
}

# Build
if (-not $SkipBuild) {
    Write-Status "Building project..."
    Push-Location $SrcPath

    $buildResult = dotnet build --configuration $Configuration 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Fail "Build failed"
        $buildResult | Write-Host
        Pop-Location
        exit 1
    }

    Pop-Location
    Write-Success "Build completed"
}

# Pack
Write-Status "Creating package..."
Push-Location $SrcPath

$packResult = dotnet pack --configuration $Configuration 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Fail "Pack failed"
    $packResult | Write-Host
    Pop-Location
    exit 1
}

Pop-Location
Write-Success "Package created"

# Show installation command
if ($newVersion) {
    Write-Host ""
    Write-Host "To update your tool installation:" -ForegroundColor Yellow
    Write-Host "dotnet tool uninstall --global firefly" -ForegroundColor White
    Write-Host "dotnet tool install --global --add-source $SrcPath/nupkg firefly --version $newVersion" -ForegroundColor White
}
