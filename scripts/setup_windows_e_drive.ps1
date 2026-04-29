$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$localRoot = Join-Path $repoRoot ".local"
$cacheRoot = Join-Path $localRoot "cache"
$tmpRoot = Join-Path $localRoot "tmp"
$stateRoot = Join-Path $localRoot "state"
$homeRoot = Join-Path $localRoot "home"
$userProfileRoot = Join-Path $localRoot "userprofile"
$userProfileTemp = Join-Path $userProfileRoot "AppData\\Local\\Temp"
$appDataRoot = Join-Path $localRoot "appdata"
$localAppDataRoot = Join-Path $localRoot "localappdata"
$venvPath = Join-Path $repoRoot ".venv"
$localPlanetariumPath = Join-Path $repoRoot ".vendor\\planetarium"

$dirs = @(
    $localRoot,
    $cacheRoot,
    $tmpRoot,
    $stateRoot,
    $homeRoot,
    $userProfileRoot,
    $userProfileTemp,
    $appDataRoot,
    $localAppDataRoot,
    (Join-Path $cacheRoot "pip"),
    (Join-Path $cacheRoot "huggingface"),
    (Join-Path $cacheRoot "huggingface\\hub"),
    (Join-Path $cacheRoot "huggingface\\datasets"),
    (Join-Path $cacheRoot "huggingface\\transformers"),
    (Join-Path $cacheRoot "torch"),
    (Join-Path $cacheRoot "wandb"),
    (Join-Path $cacheRoot "matplotlib"),
    (Join-Path $cacheRoot "pycache"),
    (Join-Path $stateRoot "wandb"),
    (Join-Path $stateRoot "jupyter\\config"),
    (Join-Path $stateRoot "jupyter\\data"),
    (Join-Path $stateRoot "jupyter\\runtime"),
    (Join-Path $stateRoot "ipython")
)

foreach ($dir in $dirs) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}

$env:TMP = $userProfileTemp
$env:TEMP = $userProfileTemp
$env:TMPDIR = $userProfileTemp
$env:USERPROFILE = $userProfileRoot
$env:APPDATA = $appDataRoot
$env:LOCALAPPDATA = $localAppDataRoot
$env:PIP_CACHE_DIR = Join-Path $cacheRoot "pip"
$env:XDG_CACHE_HOME = $cacheRoot
$env:PYTHONPATH = $repoRoot
$env:HF_HOME = Join-Path $cacheRoot "huggingface"
$env:HUGGINGFACE_HUB_CACHE = Join-Path $cacheRoot "huggingface\\hub"
$env:HF_DATASETS_CACHE = Join-Path $cacheRoot "huggingface\\datasets"
$env:TRANSFORMERS_CACHE = Join-Path $cacheRoot "huggingface\\transformers"
$env:TORCH_HOME = Join-Path $cacheRoot "torch"
$env:WANDB_DIR = Join-Path $stateRoot "wandb"
$env:WANDB_CACHE_DIR = Join-Path $cacheRoot "wandb"
$env:MPLCONFIGDIR = Join-Path $cacheRoot "matplotlib"
$env:JUPYTER_CONFIG_DIR = Join-Path $stateRoot "jupyter\\config"
$env:JUPYTER_DATA_DIR = Join-Path $stateRoot "jupyter\\data"
$env:JUPYTER_RUNTIME_DIR = Join-Path $stateRoot "jupyter\\runtime"
$env:IPYTHONDIR = Join-Path $stateRoot "ipython"
$env:PYTHONPYCACHEPREFIX = Join-Path $cacheRoot "pycache"
$env:HOME = $homeRoot

$basePython = (Get-Command python | Select-Object -ExpandProperty Source)
if (-not $basePython) {
    throw "Unable to resolve a base Python interpreter from PATH."
}

if (-not (Test-Path $venvPath)) {
    & $basePython -m venv --without-pip $venvPath
}

$pythonExe = Join-Path $venvPath "Scripts\\python.exe"
if (-not (Test-Path $pythonExe)) {
    throw "Virtual environment python not found at $pythonExe"
}

Write-Host "Repo root: $repoRoot"
Write-Host "Virtualenv: $venvPath"
Write-Host "Temp/cache root: $localRoot"
Write-Host ""
Write-Host "Environment variables are set for this PowerShell session."
Write-Host "Next suggested commands:"
Write-Host "  `$env:PIP_NO_INDEX = ''"
Write-Host "  & $basePython -m pip --python $pythonExe install --upgrade pip"
Write-Host "  & $basePython -m pip --python $pythonExe install -r requirements.txt"
if (Test-Path $localPlanetariumPath) {
    Write-Host "  & $basePython -m pip --python $pythonExe install $localPlanetariumPath"
} else {
    Write-Host "  & $basePython -m pip --python $pythonExe install git+https://github.com/BatsResearch/planetarium.git"
}
