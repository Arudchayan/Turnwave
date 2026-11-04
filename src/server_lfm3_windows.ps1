param(
    [ValidateSet('standard', 'research')]
    [string]$Variant = 'standard',
    [string]$Model = '',
    [string]$Projector = '',
    [string]$Runner = '',
    [ValidateRange(0, 65535)]
    [int]$GpuLayers = 99,
    [ValidateRange(1, 65535)]
    [int]$Port = 8080
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
if (-not $Runner) {
    $Runner = Join-Path $root 'models/runner/llama-b12032-win-cpu/llama-server.exe'
}
if (-not $Model) {
    if ($Variant -eq 'research') {
        throw 'Specify -Model explicitly when using a research model.'
    } else {
        $Model = Join-Path $root 'models/LFM3-Omni-2B-GGUF/LFM3-Omni-2B-Q4_K_M.gguf'
    }
}
if (-not $Projector) {
    if ($Variant -eq 'research') {
        throw 'Specify -Projector explicitly when using a custom research model.'
    }
    $Projector = Join-Path $root 'models/LFM3-Omni-2B-GGUF/mmproj-LFM3-Omni-2B-Q4_K_M.gguf'
}
foreach ($path in @($Runner, $Model, $Projector)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Missing LFM3 artifact: $path"
    }
}
Write-Host "Starting $Variant LFM3 voice model on 127.0.0.1:$Port"
& $Runner -m $Model -mm $Projector --host 127.0.0.1 --port $Port `
    -ngl $GpuLayers -np 1 -c 4096 --reasoning off --temp 0.7 --top-k 40 `
    --top-p 0.9 --min-p 0.05 --repeat-penalty 1.05
exit $LASTEXITCODE
