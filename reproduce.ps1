# One-command reproduction, PowerShell port of reproduce.sh.
#
#   .\reproduce.ps1                                  offline surrogate, 1,550 episodes
#   .\reproduce.ps1 -Full                            + the validation triad and high-power controls
#   .\reproduce.ps1 -Liar data\raw\liar -Pheme data\raw\pheme -Full
#
# Kept deliberately close to reproduce.sh: same steps, same run names, same
# outputs, so a run on Windows and a run on Linux are comparable artefacts.

[CmdletBinding()]
param(
  [string] $Liar,
  [string] $Pheme,
  [string] $Out = "runs",
  [switch] $Full,
  [string] $Python = ".venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $Python)) {
  throw "No interpreter at $Python. Create one with: python -m venv .venv; .venv\Scripts\pip install -e '.[dev]'"
}

$dataArgs = @()
if ($Liar)  { $dataArgs += @("--liar",  $Liar)  } else {
  Write-Host "== note: no -Liar given; using the labelled surrogate claim pool."
  Write-Host "         run 'nudgesim fetch-data' first for real LIAR claims."
}
if ($Pheme) { $dataArgs += @("--pheme", $Pheme) }

function Invoke-Step([string] $Label, [string[]] $Arguments) {
  Write-Host "== $Label"
  & $Python @Arguments
  if ($LASTEXITCODE -ne 0) { throw "$Label failed with exit code $LASTEXITCODE" }
}

Invoke-Step "0/6 tests (the payoff ledger gates everything downstream)" @("-m", "pytest", "tests/", "-q")
Invoke-Step "1/6 manipulation checks and design references" (@("-m", "nudgesim.cli") + $dataArgs + @("check", "--out", "$Out/checks"))
Invoke-Step "2/6 calibration gate (plan section 5.7) -- hard Go/No-Go" (@("-m", "nudgesim.cli") + $dataArgs + @("calibrate", "--out", "$Out/calibration"))
Invoke-Step "3/6 main grid: 1,550 episodes" (@("-m", "nudgesim.cli") + $dataArgs + @("run", "--out", "$Out/main", "--run-id", "main-declared"))
Invoke-Step "4/6 preregistered analysis" @("-m", "nudgesim.cli", "analyze", "--run", "$Out/main", "--out", "$Out/main/analysis")
Invoke-Step "    figures" @("-m", "analysis.figures", "$Out/main")
Invoke-Step "5/6 metered cost of running the grid on language models" (@("-m", "nudgesim.cli") + $dataArgs + @("cost", "--out", "$Out/cost/cost.json"))

if ($Full) {
  Write-Host "== 6/6 controls: the validation triad and high-power confirmation"
  $specs = @(
    @{ dgp = "strong";   name = "strong_dgp";        seeds = 30  },
    @{ dgp = "null";     name = "null_dgp";          seeds = 30  },
    @{ dgp = "declared"; name = "highpower";         seeds = 200 },
    @{ dgp = "strong";   name = "strong_highpower";  seeds = 200 },
    @{ dgp = "null";     name = "null_highpower";    seeds = 200 }
  )
  foreach ($spec in $specs) {
    Invoke-Step "    $($spec.name)" (@("-m", "nudgesim.cli") + $dataArgs + @(
      "--dgp", $spec.dgp, "run", "--out", "$Out/$($spec.name)", "--run-id", $spec.name,
      "--arms", "core", "--core-seeds", "$($spec.seeds)"))
    Invoke-Step "    $($spec.name) analysis" @("-m", "nudgesim.cli", "analyze",
      "--run", "$Out/$($spec.name)", "--out", "$Out/$($spec.name)/analysis")
  }

  Write-Host "== gate teeth check: the same gate with and without the novelty channel"
  # Swept across seeds on purpose: a single seed is not evidence either way,
  # since the asymmetry margin without the novelty channel sits on top of the
  # threshold and clears it occasionally by chance. See docs/CALIBRATION.md.
  foreach ($dgp in @("declared", "no-novelty")) {
    foreach ($seed in 1..6) {
      & $Python -m nudgesim.cli --seed $seed --dgp $dgp calibrate `
        --calibration-episodes 200 --out "$Out/gate_${dgp}_${seed}" *> $null
    }
    $verdicts = foreach ($seed in 1..6) {
      $file = "$Out/gate_${dgp}_${seed}/calibration.json"
      if (Test-Path $file) { Get-Content $file -Raw | ConvertFrom-Json }
    }
    if ($verdicts) {
      $go   = @($verdicts | Where-Object { $_.verdict -eq "GO" }).Count
      $mean = ($verdicts | ForEach-Object { $_.asymmetry.propagation_margin } |
               Measure-Object -Average).Average
      "  {0,-12} GO {1}/{2}  mean propagation margin {3:+0.0000;-0.0000}" -f $dgp, $go, $verdicts.Count, $mean
    }
  }
} else {
  Write-Host "== 6/6 skipped (pass -Full for the validation triad and high-power controls)"
}

Write-Host ""
Write-Host "done. results: $Out/main/results.parquet - $Out/main/analysis/analysis.json"
Write-Host "paper tables: $Python paper\build_tables.py"
