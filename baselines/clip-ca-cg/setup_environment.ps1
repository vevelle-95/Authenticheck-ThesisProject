param([switch]$CpuOnly)
$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
$taskPython = Join-Path $taskRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    $taskExistingPython = Join-Path $taskRoot '../Authenticheck-ThesisProject/.venv/Scripts/python.exe'
    if (Test-Path -LiteralPath $taskExistingPython) {
        & $taskExistingPython -m venv (Join-Path $taskRoot '.venv')
    } else {
        python -m venv (Join-Path $taskRoot '.venv')
    }
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the baseline environment.' }
}
$taskIndex = if ($CpuOnly) { 'https://download.pytorch.org/whl/cpu' } else { 'https://download.pytorch.org/whl/cu128' }
if (Get-Command uv -ErrorAction SilentlyContinue) {
    uv pip install --python $taskPython --index-url $taskIndex torch torchvision
    if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed.' }
    uv pip install --python $taskPython -r (Join-Path $taskRoot 'requirements.txt')
} else {
    & $taskPython -m pip install --index-url $taskIndex torch torchvision
    if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed.' }
    & $taskPython -m pip install -r (Join-Path $taskRoot 'requirements.txt')
}
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& $taskPython -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
