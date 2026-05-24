param(
    [string]$Python = ".\.venv\Scripts\python.exe",
    [string]$OutDir = "artifacts/wpu_v2_priority",
    [string[]]$Suite = @("selector-gap", "k-sweep", "distractor-sweep"),
    [int[]]$Seeds = @(11, 13, 17, 19, 23),
    [int]$Steps = 300,
    [int]$Samples = 300,
    [int]$BatchSize = 8,
    [int]$RuntimeRepeats = 10,
    [int]$HiddenDim = 512,
    [int]$Layers = 2,
    [int]$NumHeads = 8,
    [int]$WorkingSetSize = 16,
    [double]$SelectorLossWeight = 0.1
)

$ErrorActionPreference = "Stop"

function Invoke-CwsExperiment {
    param(
        [string]$Name,
        [string[]]$Models,
        [string[]]$ExtraArgs
    )

    $target = Join-Path $OutDir $Name
    New-Item -ItemType Directory -Force -Path $target | Out-Null
    Write-Host "Running WPU v2 suite: $Name"

    & $Python scripts/causal_working_set_experiment.py `
        --models $Models `
        --hidden-dim $HiddenDim `
        --num-heads $NumHeads `
        --layers $Layers `
        --working-set-size $WorkingSetSize `
        --steps $Steps `
        --samples $Samples `
        --batch-size $BatchSize `
        --runtime-repeats $RuntimeRepeats `
        --seeds $Seeds `
        --selector-loss-weight $SelectorLossWeight `
        --balanced-labels `
        --save-checkpoints `
        --device cuda `
        --out-dir $target `
        @ExtraArgs
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    $csv = Join-Path $target "n-sweep.csv"
    if (Test-Path $csv) {
        & $Python scripts/analyze_causal_working_set.py --input $csv --output (Join-Path $target "results.md")
    }
    $kCsv = Join-Path $target "k-sweep.csv"
    if (Test-Path $kCsv) {
        & $Python scripts/analyze_causal_working_set.py --input $kCsv --output (Join-Path $target "results.md")
    }
    $dCsv = Join-Path $target "distractor-sweep.csv"
    if (Test-Path $dCsv) {
        & $Python scripts/analyze_causal_working_set.py --input $dCsv --output (Join-Path $target "results.md")
    }
}

Write-Host "Checking CUDA environment..."
@'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
else:
    raise SystemExit("CUDA PyTorch is required for WPU v2 priority sweeps.")
'@ | & $Python -
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

foreach ($item in $Suite) {
    switch ($item) {
        "selector-gap" {
            Invoke-CwsExperiment `
                -Name "selector_gap" `
                -Models @("wpu-cws-frontier", "wpu-cws-oracle", "wpu-cws-learned") `
                -ExtraArgs @("--mode", "n-sweep", "--n-values", "64", "128", "256", "512", "1024", "2048", "4096", "8192", "--fixed-k", "8")
        }
        "k-sweep" {
            Invoke-CwsExperiment `
                -Name "k_sweep" `
                -Models @("wpu-cws-oracle", "wpu-cws-learned", "serialized-token", "graph-transformer") `
                -ExtraArgs @("--mode", "k-sweep", "--n-values", "4096", "--k-values", "4", "8", "16", "32", "64")
        }
        "distractor-sweep" {
            Invoke-CwsExperiment `
                -Name "distractor_sweep" `
                -Models @("wpu-cws-oracle", "wpu-cws-learned", "serialized-token", "graph-transformer") `
                -ExtraArgs @("--mode", "distractor-sweep", "--n-values", "4096", "--fixed-k", "8", "--distractor-values", "0", "8", "16", "32", "64", "128", "256")
        }
        "dense-n" {
            Invoke-CwsExperiment `
                -Name "dense_n_sweep" `
                -Models @("wpu-cws-oracle", "wpu-cws-learned", "serialized-token", "graph-transformer") `
                -ExtraArgs @("--mode", "n-sweep", "--n-values", "64", "128", "256", "512", "1024", "2048", "4096", "8192", "--fixed-k", "8")
        }
        "closed-loop" {
            $target = Join-Path $OutDir "long_horizon"
            New-Item -ItemType Directory -Force -Path $target | Out-Null
            Write-Host "Running WPU v2 suite: closed-loop"
            & $Python scripts/cws_closed_loop_rollout.py `
                --models wpu-cws-oracle wpu-cws-learned wpu-cws-indexed `
                --background-objects 4088 `
                --causal-obstacles 4 `
                --horizon 50 `
                --hidden-dim $HiddenDim `
                --num-heads $NumHeads `
                --layers $Layers `
                --working-set-size $WorkingSetSize `
                --device cuda `
                --output (Join-Path $target "closed_loop.csv")
            if ($LASTEXITCODE -ne 0) {
                exit $LASTEXITCODE
            }
        }
        "adaptive-hybrid" {
            Invoke-CwsExperiment `
                -Name "adaptive_hybrid" `
                -Models @("wpu-cws-indexed-sparse", "wpu-cws-indexed-local-dense", "wpu-cws-indexed-adaptive-hybrid", "serialized-token", "graph-transformer") `
                -ExtraArgs @("--mode", "k-sweep", "--n-values", "4096", "--k-values", "4", "8", "16", "32", "64", "--pre-tensor-indexed")
        }
        default {
            throw "Unknown suite '$item'. Use selector-gap, k-sweep, distractor-sweep, dense-n, closed-loop, or adaptive-hybrid."
        }
    }
}

Write-Host "WPU v2 priority sweeps complete."
