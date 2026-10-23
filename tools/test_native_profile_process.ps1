# Exercise the launch helper with ordinary Python only; never invoke Nsight or load a model.
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot 'native_profile_process.ps1')
$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).ProviderPath
$python = if ($env:JIEQI_PYTHON) { $env:JIEQI_PYTHON } else { (Get-Command python -ErrorAction Stop).Source }
$parent = [Environment]::GetEnvironmentVariables('Process')
$plain = New-NativeProfileProcessInfo -Executable $python -Arguments @('-c', "print('_helper_ok_')") -WorkingDirectory $root
$clean = New-NativeProfileProcessInfo -Executable $python -Arguments @('-c', "print('_helper_ok_')") -WorkingDirectory $root -RemovePSModulePath
if ($clean.EnvironmentVariables.ContainsKey('PSModulePath')) { throw 'PSModulePath remained in the child' }
foreach ($key in $parent.Keys) {
    if ($key -ne 'PSModulePath' -and $clean.EnvironmentVariables[$key] -cne $parent[$key]) {
        throw "Unexpected child environment change: $key"
    }
    if ([Environment]::GetEnvironmentVariable($key, 'Process') -cne $parent[$key]) {
        throw "Parent environment changed: $key"
    }
}
if ($plain.EnvironmentVariables['PSModulePath'] -cne $parent['PSModulePath']) { throw 'Plain environment changed' }
if ($clean.UseShellExecute -or -not $clean.CreateNoWindow -or $clean.WindowStyle -ne 'Hidden') { throw 'Child is not hidden' }
if (Test-NativeProfileFailure -ExitCode 0 -StandardError '') { throw 'Empty stderr incorrectly failed' }
if (-not (Test-NativeProfileFailure -ExitCode 1 -StandardError '')) { throw 'Nonzero exit was accepted' }
if (-not (Test-NativeProfileFailure -ExitCode 0 -StandardError 'std::exception::what: bad conversion')) { throw 'Known exception was accepted' }
if (Test-NativeProfileFailure -ExitCode 0 -StandardError 'Collecting data...') { throw 'Normal status incorrectly failed' }
$rejected = $false
try { [void](New-NativeProfileProcessInfo -Executable $python -Arguments @('bad"quote') -WorkingDirectory $root) } catch { $rejected = $true }
if (-not $rejected) { throw 'Unsafe quoted argument was accepted' }
$directory = Join-Path $root ('data/pilots/native_process_helper_test_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
if (Test-Path -LiteralPath $directory) { throw "Refusing to overwrite: $directory" }
[void](New-Item -ItemType Directory -Path $directory)
$marker = Join-Path $directory 'python_started.marker'
$arguments = @('-c', ("import pathlib; pathlib.Path(r'" + $marker + "').touch(exist_ok=False); print('_helper_ok_')"))
$result = Invoke-NativeProfileProcess -Executable $python -Arguments $arguments -WorkingDirectory $root `
    -LogPrefix (Join-Path $directory 'python') -RemovePSModulePath
if ($result.failed -or -not (Test-Path -LiteralPath $marker -PathType Leaf) -or $result.stdout -notmatch '(?m)^_helper_ok_\r?$') {
    throw 'Actual ordinary Python launch/marker/redirect test failed'
}
$checks = @('environment', 'exception_detection', 'argument_guard', 'actual_launch')

$livePrefix = Join-Path $directory 'live_log'
$liveMarker = Join-Path $directory 'live_log.marker'
$code = "import pathlib, time; print('_live_', flush=True); time.sleep(1); assert '_live_' in pathlib.Path(r'$livePrefix.stdout.txt').read_text(); pathlib.Path(r'$liveMarker').touch(exist_ok=False)"
$live = Invoke-NativeProfileProcess -Executable $python -Arguments @('-c',$code) -WorkingDirectory $root -LogPrefix $livePrefix
if ($live.failed -or -not (Test-Path -LiteralPath $liveMarker)) { throw 'Log was not readable before process exit' }
$checks += 'live_log'

$dual = Invoke-NativeProfileProcess -Executable $python -Arguments @('-c', "import sys; sys.stdout.write('o'*20000); sys.stderr.write('e'*20000)") `
    -WorkingDirectory $root -LogPrefix (Join-Path $directory 'dual_pipe')
if ($dual.failed -or $dual.stdout.Length -ne 20000 -or $dual.stderr.Length -ne 20000) { throw 'Concurrent pipe drain lost output' }
$checks += 'dual_pipe'

foreach ($exitCode in @(0,23)) {
    $prefix = Join-Path $directory ('inherited_pipe_' + $exitCode)
    $childPath = $prefix + '.child_pid.txt'
    $code = "import pathlib, subprocess, sys; child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(20)']); pathlib.Path(r'$childPath').write_text(str(child.pid)); print('_before_exit_', flush=True); sys.exit($exitCode)"
    $child = $null
    try {
        $timer = [Diagnostics.Stopwatch]::StartNew()
        $held = Invoke-NativeProfileProcess -Executable $python -Arguments @('-c',$code) -WorkingDirectory $root `
            -LogPrefix $prefix -DrainTimeoutSeconds 1
        $child = Get-Process -Id ([int](Get-Content -LiteralPath $childPath -Raw)) -ErrorAction Stop
        if (-not $held.failed -or $held.timed_out -or $held.pipes_complete -or $held.exit_code -ne $exitCode `
                -or $held.stdout -notmatch '_before_exit_' -or $timer.Elapsed.TotalSeconds -gt 8 -or $child.HasExited) {
            throw 'Exited parent with inherited pipes did not return a bounded, logged failure'
        }
        $checks += ('inherited_pipe_exit_' + $exitCode)
    } finally {
        # Only the ordinary Python child created by this test is stopped, never a named process set.
        if ($null -eq $child -and (Test-Path -LiteralPath $childPath)) {
            $child = Get-Process -Id ([int](Get-Content -LiteralPath $childPath -Raw)) -ErrorAction SilentlyContinue
        }
        if ($null -ne $child -and -not $child.HasExited) { $child.Kill(); [void]$child.WaitForExit(5000) }
    }
}

$timed = $null
try {
    $timer = [Diagnostics.Stopwatch]::StartNew()
    $timeout = Invoke-NativeProfileProcess -Executable $python -Arguments @('-c', "import time; print('_still_alive_', flush=True); time.sleep(20)") `
        -WorkingDirectory $root -LogPrefix (Join-Path $directory 'timeout') -TimeoutSeconds 1 -DrainTimeoutSeconds 1
    $timed = Get-Process -Id $timeout.pid -ErrorAction Stop
    if (-not $timeout.failed -or -not $timeout.timed_out -or $null -ne $timeout.exit_code -or $timeout.pipes_complete `
            -or $timeout.stdout -notmatch '_still_alive_' -or $timer.Elapsed.TotalSeconds -gt 8 -or $timed.HasExited) {
        throw 'Live process timeout did not return without killing the target'
    }
    $checks += 'timeout_no_kill'
} finally {
    if ($null -ne $timed -and -not $timed.HasExited) { $timed.Kill(); [void]$timed.WaitForExit(5000) }
}

$blockedMarker = Join-Path $directory 'should_not_start.marker'
$blocked = $false
try {
    [void](Invoke-NativeProfileProcess -Executable $python -Arguments @('-c', "import pathlib; pathlib.Path(r'$blockedMarker').touch()") `
        -WorkingDirectory $root -LogPrefix (Join-Path $directory 'python'))
} catch { $blocked = $_.Exception.Message -match 'Refusing to overwrite' }
if (-not $blocked -or (Test-Path -LiteralPath $blockedMarker)) { throw 'Existing logs were overwritten or target launched' }
$checks += 'no_overwrite'

# Evaluate only the local script's four argument-building assignments, never its admin/launch body.
$tokens = $null
$errors = $null
$captureAst = [Management.Automation.Language.Parser]::ParseFile(
    (Join-Path $PSScriptRoot 'capture_native_selfplay.ps1'), [ref]$tokens, [ref]$errors)
if ($errors.Count) { throw 'Capture script parse failed' }
$assignments = @($captureAst.EndBlock.Statements | Where-Object {
    $_ -is [Management.Automation.Language.AssignmentStatementAst] `
        -and $_.Left.Extent.Text -in @('$trace','$sample','$backtrace','$arguments')
})
if ($assignments.Count -ne 4) { throw 'Capture argument-builder changed; update its targeted test' }
$report = 'unused-report'
$workload = 'unused-workload'
$referencePath = 'unused-reference'
foreach ($TraceMode in @('cuda','cpu','combined')) {
    foreach ($assignment in $assignments) { . ([scriptblock]::Create($assignment.Extent.Text)) }
    $expectedTrace = if ($TraceMode -eq 'cpu') { '--trace=nvtx' } else { '--trace=cuda,nvtx' }
    $expectedSample = if ($TraceMode -eq 'cuda') { 'none' } else { 'process-tree' }
    $expectedBacktrace = if ($TraceMode -eq 'cuda') { '--backtrace=none' } else { '--backtrace=auto' }
    if ($arguments -notcontains $expectedTrace -or $arguments -notcontains ('--sample=' + $expectedSample) `
            -or $arguments -notcontains ('--cpuctxsw=' + $expectedSample) -or $arguments -notcontains $expectedBacktrace `
            -or $arguments -contains '--export=sqlite' -or $arguments -notcontains '--kill=false' `
            -or $arguments -notcontains '--duration=180' -or $arguments -contains '--capture-range=nvtx' `
            -or $arguments -contains '--capture-range-end=stop') { throw "Wrong isolated capture flags: $TraceMode" }
    $checks += ('capture_flags_' + $TraceMode)
}
[IO.File]::WriteAllText((Join-Path $directory 'result.json'),
    ([ordered]@{checks=$checks; passed=$true; nsys_invoked=$false; models_loaded=$false; output=$directory} | ConvertTo-Json -Depth 3),
    [Text.UTF8Encoding]::new($false))
Write-Host "Launch-helper tests passed: $($checks.Count) checks; no Nsight/training/GPU workload. Output: $directory"
