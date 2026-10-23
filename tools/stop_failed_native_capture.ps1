# Read-only by default; explicit administrator cleanup is restricted to verified frozen-workload processes.
param([Parameter(Mandatory=$true)][string]$CaptureDirectory, [switch]$StopProcesses)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Get-VerifiedNativeCaptureTargets {
    param($Capture, $Launch, $Start, [int[]]$WorkerIds, [object[]]$Snapshot, [string]$Directory,
        [string]$Python)
    if (-not $Python) { throw 'Python executable path is required for identity checks.' }
    if (-not $Capture.process_failed -or -not $Launch.failed) { throw 'Capture is not recorded as failed.' }
    if (@($Snapshot | Where-Object ProcessId -eq $Launch.pid).Count) { throw 'CLI PID is still present; do not clean an active or ambiguous session.' }
    $rootId = [int]$Start.pid
    $workers = @($WorkerIds | Sort-Object -Unique)
    if ($workers.Count -ne $Start.workers -or $rootId -in $workers) { throw 'Incomplete or ambiguous worker identity.' }
    $rootProcess = @($Snapshot | Where-Object ProcessId -eq $rootId)
    $workerProcesses = @($Snapshot | Where-Object { $_.ProcessId -in $workers })
    if ($rootProcess.Count -eq 0) {
        if ($workerProcesses.Count) { throw 'Root has exited; remaining workers need separate identity verification.' }
        return
    }
    $rootProcess = $rootProcess[0]
    $program = Join-Path $Capture.workspace 'tools\profile_native_selfplay.py'
    $workload = Join-Path $Directory 'workload'
    $startTime = ([datetime]$Launch.started_utc).ToUniversalTime()
    $rootTime = $rootProcess.CreationDate.ToUniversalTime()
    if ($rootProcess.Name -ne 'python.exe' -or $rootProcess.ExecutablePath -ne $Python `
            -or $rootProcess.ParentProcessId -ne $Launch.pid -or $rootTime -lt $startTime `
            -or $rootTime -gt $startTime.AddMinutes(2) -or -not $rootProcess.CommandLine `
            -or $rootProcess.CommandLine.IndexOf($program, [StringComparison]::OrdinalIgnoreCase) -lt 0 `
            -or $rootProcess.CommandLine.IndexOf($workload, [StringComparison]::OrdinalIgnoreCase) -lt 0) {
        throw 'Root path/command/parent/start time did not match the failed frozen workload.'
    }
    if (@($Snapshot | Where-Object { $_.ParentProcessId -eq $rootId -and $_.ProcessId -notin $workers }).Count) {
        throw 'Unrecorded root child exists; refusing process-tree cleanup.'
    }
    foreach ($worker in $workerProcesses) {
        $created = $worker.CreationDate.ToUniversalTime()
        if ($worker.Name -ne 'python.exe' -or $worker.ExecutablePath -ne $Python `
                -or $worker.ParentProcessId -ne $rootId -or $created -lt $rootTime `
                -or $created -gt $startTime.AddMinutes(2) -or $worker.CommandLine -notmatch '\bspawn_main\b' `
                -or $worker.CommandLine -notmatch ("\bparent_pid\s*=\s*" + $rootId + "\b")) {
            throw "Worker identity did not match the captured spawn process: $($worker.ProcessId)"
        }
        if (@($Snapshot | Where-Object ParentProcessId -eq $worker.ProcessId).Count) {
            throw "Unrecorded worker child exists: $($worker.ProcessId)"
        }
    }
    return @($workerProcesses) + @($rootProcess)
}

$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).ProviderPath
$pilots = Join-Path $root 'data\pilots'
$directory = (Resolve-Path -LiteralPath $CaptureDirectory).ProviderPath
if (-not $directory.StartsWith($pilots + '\', [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Capture directory must be below this workspace data/pilots.'
}
$capture = Get-Content -LiteralPath (Join-Path $directory 'capture.json') -Raw | ConvertFrom-Json
$launch = Get-Content -LiteralPath (Join-Path $directory 'nsys.process.json') -Raw | ConvertFrom-Json
if ($capture.workspace -ne $root -or $capture.output -ne $directory -or -not $capture.process_failed -or -not $launch.failed) {
    throw 'Require a failed capture from this workspace, with CLI PID metadata.'
}
$events = @(Get-Content -LiteralPath (Join-Path $directory 'workload\events.jsonl') | ForEach-Object { $_ | ConvertFrom-Json })
$starts = @($events | Where-Object event -eq 'start')
if ($starts.Count -ne 1 -or $starts[0].value.data.kind -ne 'native_profile_capture') { throw 'Missing frozen-workload identity.' }
$start = $starts[0].value.data
$workerIds = @($events | Where-Object event -eq 'warm' | ForEach-Object { $_.value.data.pids } | Sort-Object -Unique)
$snapshot = @(Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,Name,CreationDate,ExecutablePath,CommandLine)
$principal = [Security.Principal.WindowsPrincipal]::new([Security.Principal.WindowsIdentity]::GetCurrent())
$administrator = $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
$diagnosis = [ordered]@{utc=(Get-Date).ToUniversalTime().ToString('o'); capture=$directory; administrator=$administrator
    stop_requested=[bool]$StopProcesses; processes_terminated=@(); stop_errors=@()
    processes=@($snapshot | Where-Object { $_.ProcessId -in (@([int]$start.pid,[int]$launch.pid) + $workerIds) `
        -or $_.ParentProcessId -eq $start.pid })
    crash_events=@(Get-WinEvent -FilterHashtable @{LogName='Application';StartTime=([datetime]$capture.utc);Id=1000,1001} `
        -ErrorAction SilentlyContinue | Where-Object { $_.Message -match 'nsys.exe' } | Select-Object TimeCreated,Id,ProviderName,Message)}
$path = Join-Path $directory ('failure_diagnostic_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff') + '.json')
if (Test-Path -LiteralPath $path) { throw "Refusing to overwrite: $path" }
$utf8 = [Text.UTF8Encoding]::new($false)
[IO.File]::WriteAllText($path, ($diagnosis | ConvertTo-Json -Depth 7), $utf8)
if (-not $StopProcesses) {
    Write-Host "Read-only diagnosis saved; no process was stopped: $path"
    $diagnosis.processes | Format-Table ProcessId,ParentProcessId,Name,CreationDate -AutoSize
    return
}
if (-not $administrator) { throw "Diagnosis saved. Cleanup requires the original administrator terminal: $path" }
$python = if ($env:JIEQI_PYTHON) { $env:JIEQI_PYTHON } else { (Get-Command python -ErrorAction Stop).Source }
$targets = @(Get-VerifiedNativeCaptureTargets -Capture $capture -Launch $launch -Start $start `
    -WorkerIds $workerIds -Snapshot $snapshot -Directory $directory -Python $python)
# Acquire and recheck all process objects before any termination, without a broad name/tree kill.
$handles = @(foreach ($target in $targets) {
    $process = Get-Process -Id $target.ProcessId -ErrorAction SilentlyContinue
    if ($null -eq $process -or $process.HasExited) { continue }
    if ($process.ProcessName -ne 'python' `
            -or [Math]::Abs(($process.StartTime.ToUniversalTime() - $target.CreationDate.ToUniversalTime()).TotalMilliseconds) -gt 1) {
        throw "PID identity changed before cleanup: $($target.ProcessId)"
    }
    $process
})
foreach ($process in $handles) {
    try {
        if (-not $process.HasExited) {
            $process.Kill()
            if (-not $process.WaitForExit(5000)) { throw 'Process did not exit within 5 seconds.' }
            $diagnosis.processes_terminated += $process.Id
        }
    } catch { $diagnosis.stop_errors += "PID $($process.Id): $($_.Exception.Message)" }
    finally { $process.Dispose() }
}
[IO.File]::WriteAllText($path, ($diagnosis | ConvertTo-Json -Depth 7), $utf8)
if ($diagnosis.stop_errors.Count) { throw "Cleanup had errors; do not retry sampling. Inspect: $path" }
Write-Host "Verified capture processes stopped: $($diagnosis.processes_terminated -join ', '). Evidence: $path"
