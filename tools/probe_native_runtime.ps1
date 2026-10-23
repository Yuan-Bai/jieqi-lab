# Pinned new-tool compatibility probes only; no model, game, training or target termination.
param([ValidateSet('marker','single','spawn')][string]$Stage='marker',
    [ValidateSet('triggered','lifecycle')][string]$CaptureScope='triggered',
    [ValidateSet('cuda','cpu','combined')][string]$TraceMode='cuda')
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'native_profile_process.ps1')
$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).ProviderPath
$nsys = if ($env:JIEQI_NSYS) { $env:JIEQI_NSYS } else { (Get-Command nsys -ErrorAction Stop).Source }
$python = if ($env:JIEQI_PYTHON) { $env:JIEQI_PYTHON } else { (Get-Command python -ErrorAction Stop).Source }
$target = Join-Path $PSScriptRoot 'probe_native_runtime.py'
$principal = [Security.Principal.WindowsPrincipal]::new([Security.Principal.WindowsIdentity]::GetCurrent())
if ($TraceMode -ne 'cuda' -and (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator) -or $Stage -eq 'marker')) {
    throw 'CPU modes require a single/spawn probe in an existing administrator terminal.'
}
if ((Get-AuthenticodeSignature -LiteralPath $nsys).Status -ne 'Valid' `
        -or (Get-FileHash -LiteralPath $nsys -Algorithm SHA256).Hash.ToLowerInvariant() `
        -ne 'e597347ef6cb45456612c0c9593401bb6d1b85497bb18534968f309955831291') {
    throw 'New Nsight signature/hash differs from the verified local capabilities.'
}
if (@(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^(python|pythonw|nsys)\.exe$' }).Count) {
    throw 'Python/Nsight already running. Do not overlap probes or training.'
}
$wprStatus = & (Join-Path $env:SystemRoot 'System32/wpr.exe') -status 2>&1
if ($LASTEXITCODE -ne 0 -or ($wprStatus -join "`n") -notmatch 'WPR is not recording') {
    throw 'WPR status is not clear; no probe was started.'
}
$directory = Join-Path $root ('data/pilots/native_2026_' + $Stage + '_' + $CaptureScope + '_' + $TraceMode + '_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
if (Test-Path -LiteralPath $directory) { throw "Refusing to overwrite: $directory" }
[void](New-Item -ItemType Directory -Path $directory)
$report = Join-Path $directory 'native'
$workload = Join-Path $directory 'workload'
$trace = if ($Stage -eq 'marker') { 'none' } elseif ($TraceMode -eq 'cpu') { 'nvtx' } else { 'cuda,nvtx' }
$sample = if ($TraceMode -eq 'cuda') { 'none' } else { 'process-tree' }
$backtrace = if ($TraceMode -eq 'cuda') { 'none' } else { 'auto' }
$arguments = @('profile',('--trace=' + $trace),('--sample=' + $sample),('--cpuctxsw=' + $sample),('--backtrace=' + $backtrace),
    '--duration=30','--kill=false','--wait=all','--stop-on-exit=true','--force-overwrite=false',
    '--resolve-symbols=false','--env-var=NSYS_NVTX_PROFILER_REGISTER_ONLY=0',('--output=' + $report))
if ($Stage -ne 'marker' -and $CaptureScope -eq 'triggered') {
    $arguments += @('--capture-range=nvtx','--nvtx-capture=jieqi_native_window','--capture-range-end=stop')
}
$arguments += @($python,$target,'--mode',$Stage,'--output-dir',$workload)
$metadata = [ordered]@{kind='native_runtime_compatibility'; stage=$Stage; trace_mode=$TraceMode; capture_scope=$CaptureScope; workspace=$root; output=$directory
    utc=(Get-Date).ToUniversalTime().ToString('o'); administrator=$principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    nsys=$nsys; nsys_sha256=(Get-FileHash -LiteralPath $nsys -Algorithm SHA256).Hash.ToLowerInvariant()
    arguments=$arguments; models_loaded=$false; training_started=$false; target_killed=$false; coverage_verified=$false}
$metadata.source_hashes = [ordered]@{}
foreach ($file in @($target,$python,(Join-Path $PSScriptRoot 'probe_native_runtime.ps1'),
        (Join-Path $PSScriptRoot 'native_profile_process.ps1'))) {
    $metadata.source_hashes[$file] = (Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLowerInvariant()
}
$utf8 = [Text.UTF8Encoding]::new($false)
$path = Join-Path $directory 'probe.json'
[IO.File]::WriteAllText($path, ($metadata | ConvertTo-Json -Depth 6), $utf8)
Write-Host "Starting synthetic compatibility stage $Stage. No model/training. Output: $directory"
$profile = Invoke-NativeProfileProcess -Executable $nsys -Arguments $arguments -WorkingDirectory $root `
    -LogPrefix (Join-Path $directory 'nsys') -RemovePSModulePath -TimeoutSeconds 90
$metadata.profile = [ordered]@{pid=$profile.pid; exit_code=$profile.exit_code; failed=$profile.failed
    timed_out=$profile.timed_out; pipes_complete=$profile.pipes_complete}
[IO.File]::WriteAllText($path, ($metadata | ConvertTo-Json -Depth 6), $utf8)
if ($profile.failed) { throw "Synthetic profile failed; retain logs/PIDs, no automatic retry/termination: $directory" }
$resultPath = Join-Path $workload 'result.json'
if (-not (Test-Path -LiteralPath $resultPath) -or -not (Test-Path -LiteralPath ($report + '.nsys-rep'))) {
    throw "Workload or report missing: $directory"
}
$result = Get-Content -LiteralPath $resultPath -Raw | ConvertFrom-Json
$expectedRows = if ($Stage -eq 'marker') { 0 } elseif ($Stage -eq 'single') { 1 } else { 2 }
if ($result.kind -ne 'native_runtime_probe' -or $result.mode -ne $Stage -or $result.models_loaded `
        -or $result.training_started -or @($result.rows).Count -ne $expectedRows `
        -or @($result.rows | Where-Object { -not $_.exact -or $_.dtype -ne 'torch.float32' `
            -or $_.tf32_matmul -or -not $_.tf32_cudnn }).Count) { throw 'Synthetic workload gate failed' }
if ($Stage -eq 'marker' -and -not (Test-Path -LiteralPath (Join-Path $workload 'python_started.marker'))) {
    throw 'Actual Python marker missing'
}
$export = Invoke-NativeProfileProcess -Executable $nsys `
    -Arguments @('export','--type=sqlite',('--output=' + $report + '.sqlite'),'--force-overwrite=false',($report + '.nsys-rep')) `
    -WorkingDirectory $root -LogPrefix (Join-Path $directory 'sqlite_export') -RemovePSModulePath -TimeoutSeconds 60
$metadata.export = [ordered]@{pid=$export.pid;exit_code=$export.exit_code;failed=$export.failed
    timed_out=$export.timed_out;pipes_complete=$export.pipes_complete}
foreach ($file in $metadata.source_hashes.Keys) {
    if ((Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLowerInvariant() -ne $metadata.source_hashes[$file]) {
        throw "Probe source changed: $file"
    }
}
$metadata.workload_and_cli_complete = -not $export.failed
[IO.File]::WriteAllText($path, ($metadata | ConvertTo-Json -Depth 6), $utf8)
if ($export.failed -or -not (Test-Path -LiteralPath ($report + '.sqlite'))) { throw "Probe export failed: $directory" }
Write-Host "Synthetic stage $Stage completed; inspect SQLite event/PID coverage before advancing. Output: $directory"
