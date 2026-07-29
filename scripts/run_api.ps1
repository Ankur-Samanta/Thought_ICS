# 1. 接收用户参数
# 2. 检查 API 密钥
# 3. 找到 Python 3.12 环境
# 4. 拼出并执行 batch_eval.py 的命令

# [CmdletBinding()]:把这个普通脚本当作“高级 PowerShell 命令”处理
[CmdletBinding()]
param(
    [ValidateSet(1, 2, 3)]
    [int]$AutonomyLevel = 3,

    [ValidateSet("math500", "gsm8k", "amc23", "aime", "csqa", "gpqa", "svamp", "mathqa", "imo", "imobench")]
    [string]$Dataset = "amc23",

    [int]$NProblems = 1,  # 表示要运行多少道题。
    [int]$MaxIterations = 2,
    [string]$Model = $env:OPENAI_MODEL,
    [string]$BaseUrl = $env:OPENAI_BASE_URL,
    [string]$PythonExecutable = ""  # 使用哪个 Python 解释器
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($env:OPENAI_API_KEY)) {
    throw "Set OPENAI_API_KEY in this PowerShell session before running the API evaluation."
}

if ([string]::IsNullOrWhiteSpace($Model)) {
    $Model = "Qwen/Qwen3-14B"
}

if ([string]::IsNullOrWhiteSpace($PythonExecutable)) {
    $workspaceRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    $managedPython = Join-Path $workspaceRoot ".conda_envs\Thought_ICS\python.exe"
    if (Test-Path -LiteralPath $managedPython) {
        $PythonExecutable = $managedPython
    } else {
        $PythonExecutable = "python"
    }
}

# 建立参数数组
$arguments = @(
    "-m", "thought_ics.eval.batch_eval",
    "--3p",
    "--autonomy-level", $AutonomyLevel,
    "--dataset", $Dataset,
    "--n-problems", $NProblems,
    "--max-iterations", $MaxIterations,
    "--3p-model", $Model
)

if (-not [string]::IsNullOrWhiteSpace($BaseUrl)) {
    $arguments += @("--3p-base-url", $BaseUrl)
}

# & 是 PowerShell 的调用运算符
& $PythonExecutable @arguments
exit $LASTEXITCODE
