param(
    [string]$Python = "python",
    [string]$OutDir = "artifacts/causal_working_set_8m_gpu",
    [int[]]$NValues = @(64, 128, 256, 512, 1024, 2048, 4096),
    [int[]]$Seeds = @(11, 13, 17, 19, 23),
    [int]$Steps = 500,
    [int]$Samples = 512,
    [int]$BatchSize = 8,
    [int]$RuntimeRepeats = 30,
    [double]$SelectorLossWeight = 0.1
)

$ErrorActionPreference = "Stop"

Write-Host "Checking PyTorch/CUDA environment..."
@'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    print("memory_gb", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 3))
else:
    raise SystemExit("CUDA PyTorch is required for the 8M GPU run.")
'@ | & $Python -
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$nArgs = $NValues | ForEach-Object { "$_" }
$seedArgs = $Seeds | ForEach-Object { "$_" }

& $Python scripts/causal_working_set_experiment.py `
    --models wpu-cws-frontier wpu-cws-oracle wpu-cws-learned serialized-token graph-transformer `
    --n-values $nArgs `
    --fixed-k 8 `
    --hidden-dim 512 `
    --num-heads 8 `
    --layers 2 `
    --working-set-size 16 `
    --steps $Steps `
    --samples $Samples `
    --batch-size $BatchSize `
    --runtime-repeats $RuntimeRepeats `
    --seeds $seedArgs `
    --selector-loss-weight $SelectorLossWeight `
    --device cuda `
    --out-dir $OutDir
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& $Python scripts/analyze_causal_working_set.py `
    --input "$OutDir/n-sweep.csv" `
    --output "docs/experiments/causal_working_set_8m_gpu_results.md"
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "CWS 8M GPU run complete."
