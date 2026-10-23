# Frozen sampler. CPU modes require an existing admin terminal; CUDA-only does not.
# No production training, policy changes or process termination.
param([Parameter(Mandatory=$true)][string]$ReferenceFile,
    [Parameter(Mandatory=$true)][string]$CompatibilityFile,
    [ValidateSet('combined','cuda','cpu')][string]$TraceMode='combined')
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$principal = [Security.Principal.WindowsPrincipal]::new([Security.Principal.WindowsIdentity]::GetCurrent())
if ($TraceMode -ne 'cuda' -and -not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Run this script in the existing administrator PowerShell terminal.'
}
. (Join-Path $PSScriptRoot 'native_profile_process.ps1')
$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).ProviderPath
$pilots = (Resolve-Path -LiteralPath (Join-Path $root 'data/pilots')).ProviderPath
$referencePath = (Resolve-Path -LiteralPath $ReferenceFile).ProviderPath
if (-not $referencePath.StartsWith($pilots + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Reference must be a completed preparation under this workspace data/pilots.'
}
$reference = Get-Content -LiteralPath $referencePath -Raw | ConvertFrom-Json
if ($reference.kind -ne 'native_profile_preparation' -or -not $reference.production_unchanged `
        -or -not $reference.paired_roots_exact -or $reference.repeats -ne 4 `
        -or @($reference.graph_checks).Count -ne 36 `
        -or @($reference.graph_checks | Where-Object { -not $_.exact }).Count -ne 0) {
    throw 'Reference has not passed the required graph/network/policy/RNG gates.'
}
foreach ($property in $reference.source_hashes.PSObject.Properties) {
    if ((Get-FileHash -LiteralPath $property.Name -Algorithm SHA256).Hash.ToLowerInvariant() -ne $property.Value) {
        throw "Prepared source changed: $($property.Name). Repeat the normal-user preparation."
    }
}
$python = if ($env:JIEQI_PYTHON) { $env:JIEQI_PYTHON } else { (Get-Command python -ErrorAction Stop).Source }
if ($reference.python -ne $python) { throw 'Prepared Python runtime differs from the fixed local interpreter.' }
$nsys = if ($env:JIEQI_NSYS) { $env:JIEQI_NSYS } else { (Get-Command nsys -ErrorAction Stop).Source }
if ((Get-AuthenticodeSignature -LiteralPath $nsys).Status -ne 'Valid') { throw 'Cannot verify installed Nsight signature.' }
if ((Get-FileHash -LiteralPath $nsys -Algorithm SHA256).Hash.ToLowerInvariant() `
        -ne 'e597347ef6cb45456612c0c9593401bb6d1b85497bb18534968f309955831291') {
    throw 'Nsight binary differs from the verified 2026.5.1 compatibility probes.'
}
$compatibilityPath = (Resolve-Path -LiteralPath $CompatibilityFile).ProviderPath
if (-not $compatibilityPath.StartsWith($pilots + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Compatibility evidence must be under this workspace data/pilots.'
}
$compatibility = Get-Content -LiteralPath $compatibilityPath -Raw | ConvertFrom-Json
$compatibilityDirectory = Split-Path -Parent $compatibilityPath
$probe = Get-Content -LiteralPath (Join-Path $compatibilityDirectory 'probe.json') -Raw | ConvertFrom-Json
if ($compatibility.kind -ne 'native_runtime_coverage' -or $compatibility.mode -ne 'spawn' `
        -or $compatibility.capture_scope -ne 'lifecycle' -or -not $compatibility.known_workload_coverage_passed `
        -or @($compatibility.per_pid).Count -ne 2 -or @($compatibility.errors).Count -ne 0 `
        -or $probe.nsys -ne $nsys -or $probe.nsys_sha256 -ne (Get-FileHash -LiteralPath $nsys -Algorithm SHA256).Hash.ToLowerInvariant() `
        -or -not $probe.workload_and_cli_complete) {
    throw 'Require completed two-child lifecycle coverage with the pinned new Nsight binary.'
}
foreach ($entry in @(@('native.sqlite','sqlite_sha256'),@('workload/result.json','workload_sha256'))) {
    if ((Get-FileHash -LiteralPath (Join-Path $compatibilityDirectory $entry[0]) -Algorithm SHA256).Hash.ToLowerInvariant() `
            -ne $compatibility.($entry[1])) { throw 'Compatibility artifacts changed after inspection.' }
}
$wprStatus = & (Join-Path $env:SystemRoot 'System32/wpr.exe') -status 2>&1
if ($LASTEXITCODE -ne 0 -or ($wprStatus -join "`n") -notmatch 'WPR is not recording') {
    throw 'WPR is recording or its status could not be confirmed. Do not overlap profiling sessions.'
}
if (@(Get-Process -Name python, pythonw, nsys -ErrorAction SilentlyContinue).Count -ne 0) {
    throw 'Python or Nsight is already running. Finish existing training/profiling before this isolated capture.'
}
$directory = Join-Path $pilots ('native_profile_capture_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
if (Test-Path -LiteralPath $directory) { throw "Refusing to overwrite: $directory" }
[void](New-Item -ItemType Directory -Path $directory)
$workload = Join-Path $directory 'workload'
$report = Join-Path $directory 'native'
$trace = if ($TraceMode -eq 'cpu') { 'nvtx' } else { 'cuda,nvtx' }
$sample = if ($TraceMode -eq 'cuda') { 'none' } else { 'process-tree' }
$backtrace = if ($TraceMode -eq 'cuda') { 'none' } else { 'auto' }
# A parent NVTX start missed a child's first range in the tiny probe. Collect the
# full lifecycle instead and require post-export clipping/coverage of the closed window.
$arguments = @('profile', ('--trace=' + $trace), ('--sample=' + $sample), ('--cpuctxsw=' + $sample),
    ('--backtrace=' + $backtrace), '--sampling-frequency=1000', '--duration=180', '--kill=false', '--wait=all', '--stop-on-exit=true',
    '--resolve-symbols=false', '--retain-etw-files=true', '--force-overwrite=false',
    '--env-var=JIEQI_RULES_BACKEND=cython,JIEQI_LEGAL_BACKEND=cython,JIEQI_SEARCH_BACKEND=python,NSYS_NVTX_PROFILER_REGISTER_ONLY=0',
    ('--output=' + $report), $python, (Join-Path $root 'tools/profile_native_selfplay.py'),
    '--output-dir', $workload, '--reference-file', $referencePath)
$metadata = [ordered]@{
    administrator=$principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    utc=(Get-Date).ToUniversalTime().ToString('o'); workspace=$root
    reference=$referencePath; reference_sha256=(Get-FileHash -LiteralPath $referencePath -Algorithm SHA256).Hash.ToLowerInvariant()
    compatibility=$compatibilityPath; compatibility_sha256=(Get-FileHash -LiteralPath $compatibilityPath -Algorithm SHA256).Hash.ToLowerInvariant()
    nsys=$nsys; nsys_sha256=(Get-FileHash -LiteralPath $nsys -Algorithm SHA256).Hash.ToLowerInvariant()
    arguments=$arguments; output=$directory; max_capture_seconds=180
    process_tree_only=$true; symbols_resolved=$false; kill_target=$false; exit_code=$null
    launch_timeout_seconds=60; process_timeout_seconds=300; pipe_drain_timeout_seconds=3
    trace_mode=$TraceMode; capture_scope='lifecycle'; measurement_window='jieqi_native_window'
    coverage_verified=$false; sqlite_export_separate=$true
    removed_child_environment=@('PSModulePath'); parent_environment_changed=$false
}
$utf8 = [Text.UTF8Encoding]::new($false)
$metadataPath = Join-Path $directory 'capture.json'
[IO.File]::WriteAllText($metadataPath, ($metadata | ConvertTo-Json -Depth 6), $utf8)
Write-Host "Checking an actual Python launch without tracing. Output: $directory"
$marker = Join-Path $directory 'python_started.marker'
$probeArguments = @('profile', '--trace=none', '--sample=none', '--cpuctxsw=none',
    '--kill=false', '--wait=all', '--duration=1', '--force-overwrite=false',
    ('--output=' + (Join-Path $directory 'startup_probe')), $python, '-c',
    ("import pathlib; pathlib.Path(r'" + $marker + "').touch(exist_ok=False)"))
$probeResult = Invoke-NativeProfileProcess -Executable $nsys -Arguments $probeArguments -WorkingDirectory $root `
    -LogPrefix (Join-Path $directory 'startup_probe') -RemovePSModulePath -TimeoutSeconds 60
$metadata.startup_probe = [ordered]@{exit_code=$probeResult.exit_code; failed=$probeResult.failed
    python_marker_seen=(Test-Path -LiteralPath $marker -PathType Leaf)
    timed_out=$probeResult.timed_out; pipes_complete=$probeResult.pipes_complete}
[IO.File]::WriteAllText($metadataPath, ($metadata | ConvertTo-Json -Depth 6), $utf8)
if ($probeResult.failed -or -not $metadata.startup_probe.python_marker_seen) {
    throw "Python launch check failed; no self-play was started. Inspect startup_probe logs: $directory"
}
Write-Host 'Python launch verified. Collecting lifecycle; analyze only the warmed NVTX window. No production training.'
$process = Invoke-NativeProfileProcess -Executable $nsys -Arguments $arguments -WorkingDirectory $root `
    -LogPrefix (Join-Path $directory 'nsys') -RemovePSModulePath
$metadata.exit_code = $process.exit_code
$metadata.launch_exception = [bool](($process.stdout + "`n" + $process.stderr) -match 'Unexpected exception|bad conversion|Dynamic exception type')
$metadata.process_failed = $process.failed
$metadata.process_timed_out = $process.timed_out
$metadata.pipes_complete = $process.pipes_complete
$metadata.log_errors = $process.log_errors
[IO.File]::WriteAllText($metadataPath, ($metadata | ConvertTo-Json -Depth 6), $utf8)
if ($process.failed) {
    throw "Nsight failed: exit=$($process.exit_code), timeout=$($process.timed_out), pipes_complete=$($process.pipes_complete). Logs and PID metadata were saved. No retry or process termination was performed: $directory"
}
# A capped trace must not terminate its target. Allow it to finish its four fixed cohorts.
$deadline = (Get-Date).AddMinutes(5)
$resultPath = Join-Path $workload 'result.json'
while (-not (Test-Path -LiteralPath $resultPath) -and (Get-Date) -lt $deadline) {
    if (@(Get-Process -Name python, pythonw -ErrorAction SilentlyContinue).Count -eq 0) {
        throw "Python has exited without a final workload result. No process was killed. Inspect workload/Nsight logs: $directory"
    }
    Start-Sleep -Seconds 2
}
if (-not (Test-Path -LiteralPath $resultPath)) {
    throw "Workload result is incomplete. No process was killed; inspect logs/PIDs before rerunning: $directory"
}
$result = Get-Content -LiteralPath $resultPath -Raw | ConvertFrom-Json
if (-not $result.production_unchanged -or -not $result.paired_roots_exact) { throw "Workload exact gate failed: $directory" }
if (-not (Test-Path -LiteralPath ($report + '.nsys-rep'))) {
    throw "Native report is missing. Do not assume that tracing succeeded: $directory"
}
$export = Invoke-NativeProfileProcess -Executable $nsys `
    -Arguments @('export','--type=sqlite',('--output=' + $report + '.sqlite'),'--force-overwrite=false',($report + '.nsys-rep')) `
    -WorkingDirectory $root -LogPrefix (Join-Path $directory 'sqlite_export') -RemovePSModulePath -TimeoutSeconds 120
$metadata.sqlite_export = [ordered]@{pid=$export.pid; exit_code=$export.exit_code; failed=$export.failed
    timed_out=$export.timed_out; pipes_complete=$export.pipes_complete}
[IO.File]::WriteAllText($metadataPath, ($metadata | ConvertTo-Json -Depth 6), $utf8)
if ($export.failed -or -not (Test-Path -LiteralPath ($report + '.sqlite'))) {
    throw "Separate SQLite export failed; the original report is retained. Inspect sqlite_export logs/PID: $directory"
}
Write-Host "Capture finished ($TraceMode); exact workload gates passed. Reply with: native capture complete. Output: $directory"
Write-Host 'Trace contents, child coverage, sample loss, symbols and overhead still need analysis.'
