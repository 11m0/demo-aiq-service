#requires -Version 7.0

[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RunnerArguments
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = $PSScriptRoot
$pythonExecutable = Join-Path $projectRoot ".venv\Scripts\python.exe"
$runner = Join-Path $projectRoot "run.py"

if (-not (Test-Path -LiteralPath $pythonExecutable -PathType Leaf)) {
    throw "Не найден Python проекта: $pythonExecutable"
}
if (-not (Test-Path -LiteralPath $runner -PathType Leaf)) {
    throw "Не найден Python launcher: $runner"
}

Push-Location -LiteralPath $projectRoot
try {
    & $pythonExecutable $runner @RunnerArguments
    $processExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}

exit $processExitCode
