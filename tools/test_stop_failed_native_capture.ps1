# Pure identity tests. Never enumerate or stop real processes, invoke Nsight, or load a model.
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$tokens = $null
$errors = $null
$ast = [Management.Automation.Language.Parser]::ParseFile(
    (Join-Path $PSScriptRoot 'stop_failed_native_capture.ps1'), [ref]$tokens, [ref]$errors)
if ($errors.Count) { throw 'Cleanup script parse failed' }
$function = $ast.Find({param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] `
    -and $node.Name -eq 'Get-VerifiedNativeCaptureTargets'}, $true)
. ([scriptblock]::Create($function.Extent.Text))
function New-TestFixture {
    $python = if ($env:JIEQI_PYTHON) { $env:JIEQI_PYTHON } else { (Get-Command python -ErrorAction Stop).Source }
    $directory = 'C:\fixtures\jieqi-lab\data\pilots\test_capture'
    $capture = [pscustomobject]@{workspace='C:\fixtures\jieqi-lab';process_failed=$true}
    $launch = [pscustomobject]@{pid=99;started_utc='2026-10-08T14:36:39Z';failed=$true}
    $start = [pscustomobject]@{pid=100;workers=8}
    $root = [pscustomobject]@{ProcessId=100;ParentProcessId=99;Name='python.exe'
        ExecutablePath=$python;CreationDate=[datetime]'2026-10-08T14:36:41Z'
        CommandLine=('python.exe C:\fixtures\jieqi-lab\tools\profile_native_selfplay.py --output-dir ' + $directory + '\workload')}
    $snapshot = @($root) + @(200..207 | ForEach-Object { [pscustomobject]@{ProcessId=$_;ParentProcessId=100;Name='python.exe'
        ExecutablePath=$python;CreationDate=[datetime]'2026-10-08T14:36:50Z'
        CommandLine='from multiprocessing.spawn import spawn_main; spawn_main(parent_pid=100, pipe_handle=1)'} })
    return @{Capture=$capture;Launch=$launch;Start=$start;WorkerIds=200..207;Snapshot=$snapshot;Directory=$directory;Python=$python}
}
$valid = New-TestFixture
$targets = @(Get-VerifiedNativeCaptureTargets @valid)
if ($targets.Count -ne 9 -or $targets[-1].ProcessId -ne 100) { throw 'Expected eight workers then root' }
$checks = 1
foreach ($case in @('active_cli','wrong_program','wrong_directory','wrong_parent','old_root','wrong_runtime',
        'reused_worker','wrong_worker','extra_root_child','extra_worker_child','missing_root','incomplete_warmup','not_failed')) {
    $f = New-TestFixture
    switch ($case) {
        active_cli { $f.Snapshot += [pscustomobject]@{ProcessId=99;ParentProcessId=1} }
        wrong_program { $f.Snapshot[0].CommandLine = $f.Snapshot[0].CommandLine.Replace('profile_native_selfplay.py','azero.py') }
        wrong_directory { $f.Snapshot[0].CommandLine = $f.Snapshot[0].CommandLine.Replace('test_capture','other_capture') }
        wrong_parent { $f.Snapshot[0].ParentProcessId = 1 }
        old_root { $f.Snapshot[0].CreationDate = [datetime]'2026-10-08T13:36:41Z' }
        wrong_runtime { $f.Snapshot[0].ExecutablePath = 'C:\other\python.exe' }
        reused_worker { $f.Snapshot[1].CreationDate = [datetime]'2026-10-08T15:36:50Z' }
        wrong_worker { $f.Snapshot[1].CommandLine = 'python azero.py' }
        extra_root_child { $f.Snapshot += [pscustomobject]@{ProcessId=888;ParentProcessId=100} }
        extra_worker_child { $f.Snapshot += [pscustomobject]@{ProcessId=888;ParentProcessId=200} }
        missing_root { $f.Snapshot = @($f.Snapshot | Where-Object ProcessId -ne 100) }
        incomplete_warmup { $f.WorkerIds = 200..206 }
        not_failed { $f.Capture.process_failed = $false }
    }
    $rejected = $false
    try { [void](Get-VerifiedNativeCaptureTargets @f) } catch { $rejected = $true }
    if (-not $rejected) { throw "Unsafe identity accepted: $case" }
    $checks++
}
$empty = New-TestFixture
$empty.Snapshot = @()
if (@(Get-VerifiedNativeCaptureTargets @empty).Count) { throw 'Already-exited capture returned cleanup targets' }
$checks++
Write-Host "Cleanup identity tests passed: $checks checks; no real process interaction."
