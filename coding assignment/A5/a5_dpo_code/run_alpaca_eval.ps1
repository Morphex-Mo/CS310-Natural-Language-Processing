param(
    [Parameter(Mandatory = $false)]
    [string]$WorkspaceRoot = "E:/CS310-Natural-Language-Processing",

    [Parameter(Mandatory = $false)]
    [string]$ModelCheckpoint = "E:/CS310-Natural-Language-Processing/coding assignment/A5/a5_dpo_code/gpt2-355M-dpo.pth",

    [Parameter(Mandatory = $false)]
    [string]$ModelOutputs = "E:/CS310-Natural-Language-Processing/coding assignment/A5/a5_dpo_code/model_outputs.json"
)

$ErrorActionPreference = "Stop"

$a5CodeDir = Join-Path $WorkspaceRoot "coding assignment/A5/a5_dpo_code"
$openaiConfig = Join-Path $a5CodeDir "openai_configs.yaml"
$alpacaEvalInput = Join-Path $a5CodeDir "alpaca_eval.json"
$referenceOutputs = Join-Path $a5CodeDir "reference_outputs.json"
$annotatorsConfigDir = Join-Path $WorkspaceRoot "coding assignment/A5/qwen_judge/qwen_judge"

if (!(Test-Path $ModelCheckpoint)) {
    throw "Model checkpoint not found: $ModelCheckpoint"
}
if (!(Test-Path $openaiConfig)) {
    throw "openai_configs.yaml not found: $openaiConfig"
}
if (!(Test-Path $alpacaEvalInput)) {
    throw "alpaca_eval.json not found: $alpacaEvalInput"
}
if (!(Test-Path $referenceOutputs)) {
    throw "reference_outputs.json not found: $referenceOutputs"
}
if (!(Test-Path $annotatorsConfigDir)) {
    throw "qwen_judge folder not found: $annotatorsConfigDir"
}

Write-Host "[1/2] Generating model outputs..."
python (Join-Path $a5CodeDir "generate_dpo_responses.py") `
    --input $alpacaEvalInput `
    --output $ModelOutputs `
    --model $ModelCheckpoint

if (!(Test-Path $ModelOutputs)) {
    throw "Failed to create model outputs: $ModelOutputs"
}

Write-Host "[2/2] Running alpaca_eval..."
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:OPENAI_CLIENT_CONFIG_PATH = $openaiConfig

alpaca_eval evaluate `
    --model_outputs $ModelOutputs `
    --reference_outputs $referenceOutputs `
    --annotators_config $annotatorsConfigDir

Write-Host "Done. Check leaderboard.csv and annotations.json in the alpaca_eval output directory."