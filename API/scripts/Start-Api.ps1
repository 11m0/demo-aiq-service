#requires -Version 7.0

[CmdletBinding()]
param(
    [Parameter()]
    [ValidateNotNullOrEmpty()]
    [string]$BindAddress = "127.0.0.1",

    [Parameter()]
    [ValidateRange(1, 65535)]
    [int]$Port = 8000,

    [Parameter()]
    [switch]$Reload
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Assert-LoopbackAddress {
    <#
    .SYNOPSIS
    Запрещает случайно публиковать локальный smoke-сервер во внешней сети.
    #>
    param(
        [Parameter(Mandatory)]
        [string]$Value
    )

    if ($Value -ieq "localhost") {
        return
    }

    $parsedAddress = $null
    if (
        -not [System.Net.IPAddress]::TryParse($Value, [ref]$parsedAddress) -or
        -not [System.Net.IPAddress]::IsLoopback($parsedAddress)
    ) {
        throw "BindAddress должен быть loopback-адресом."
    }
}

Assert-LoopbackAddress -Value $BindAddress

$apiDirectory = Split-Path -Parent $PSScriptRoot
$projectRoot = Split-Path -Parent $apiDirectory
$pythonExecutable = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $pythonExecutable -PathType Leaf)) {
    throw "Не найден Python проекта: $pythonExecutable"
}

$sourceDirectories = @(
    (Join-Path $projectRoot "Agent\src"),
    (Join-Path $projectRoot "API\src"),
    (Join-Path $projectRoot "Core\src"),
    (Join-Path $projectRoot "Contracts\src")
)
foreach ($sourceDirectory in $sourceDirectories) {
    if (-not (Test-Path -LiteralPath $sourceDirectory -PathType Container)) {
        throw "Не найден каталог исходного кода: $sourceDirectory"
    }
}

$uvicornArguments = @(
    "-m",
    "uvicorn",
    "questionnaire_api.main:create_app",
    "--factory",
    "--host",
    $BindAddress,
    "--port",
    $Port.ToString([System.Globalization.CultureInfo]::InvariantCulture)
)
if ($Reload) {
    $uvicornArguments += "--reload"
}

$previousPythonPath = $env:PYTHONPATH
$locationWasChanged = $false
$processExitCode = 1

try {
    $env:PYTHONPATH = [string]::Join(
        [System.IO.Path]::PathSeparator,
        $sourceDirectories
    )
    Push-Location -LiteralPath $projectRoot
    $locationWasChanged = $true

    & $pythonExecutable @uvicornArguments
    $processExitCode = $LASTEXITCODE
}
finally {
    if ($locationWasChanged) {
        Pop-Location
    }
    if ($null -eq $previousPythonPath) {
        Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    }
    else {
        $env:PYTHONPATH = $previousPythonPath
    }
}

exit $processExitCode
