# Minimal Nsight launch diagnostic: no model, CUDA imports, CPU sampling or tracing.
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$principal = [Security.Principal.WindowsPrincipal]::new([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'This installed Nsight executable requires elevation even without tracing. Use the existing administrator terminal.'
}
. (Join-Path $PSScriptRoot 'native_profile_process.ps1')
$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).ProviderPath
$directory = Join-Path $root ('data/pilots/native_launch_probe_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
if (Test-Path -LiteralPath $directory) { throw "Refusing to overwrite: $directory" }
[void](New-Item -ItemType Directory -Path $directory)
$nsys = if ($env:JIEQI_NSYS) { $env:JIEQI_NSYS } else { (Get-Command nsys -ErrorAction Stop).Source }
$python = if ($env:JIEQI_PYTHON) { $env:JIEQI_PYTHON } else { (Get-Command python -ErrorAction Stop).Source }
if ((Get-AuthenticodeSignature -LiteralPath $nsys).Status -ne 'Valid') { throw 'Cannot verify Nsight signature.' }
$originalPSModulePath = [Environment]::GetEnvironmentVariable('PSModulePath', 'Process')
$results = @()
foreach ($mode in @('inherited', 'without_psmodulepath', 'restored')) {
    $marker = Join-Path $directory ($mode + '.python_started.marker')
    $arguments = @('profile', '--trace=none', '--sample=none', '--cpuctxsw=none',
        '--kill=false', '--wait=all', '--duration=1', '--force-overwrite=false',
        ('--output=' + (Join-Path $directory $mode)), $python, '-c',
        ("import pathlib; pathlib.Path(r'" + $marker + "').touch(exist_ok=False); print('_native_launch_ok_')"))
    $process = Invoke-NativeProfileProcess -Executable $nsys -Arguments $arguments -WorkingDirectory $root `
        -LogPrefix (Join-Path $directory $mode) -RemovePSModulePath:($mode -eq 'without_psmodulepath') -TimeoutSeconds 60
    $row = [ordered]@{mode=$mode; exit_code=$process.exit_code; stdout=$process.stdout; stderr=$process.stderr
        failed=$process.failed; timed_out=$process.timed_out; pipes_complete=$process.pipes_complete
        target_marker_seen=(($process.stdout + "`n" + $process.stderr) -match '(?m)^_native_launch_ok_\r?$')
        target_file_marker_seen=(Test-Path -LiteralPath $marker -PathType Leaf)
        launch_exception=($process.stderr -match 'Unexpected exception|bad conversion|Dynamic exception type')}
    $results += $row
    $row | ConvertTo-Json -Compress
    if ($process.timed_out -or -not $process.pipes_complete) {
        throw "Launch diagnostic stalled. Keep PID/log evidence and do not launch another probe: $directory"
    }
}
$utf8 = [Text.UTF8Encoding]::new($false)
$result = [ordered]@{
    directory=$directory
    nsys_sha256=(Get-FileHash -LiteralPath $nsys -Algorithm SHA256).Hash.ToLowerInvariant()
    non_ascii_environment_names=@(Get-ChildItem Env: | Where-Object { $_.Name -match '[^\x00-\x7F]' -or $_.Value -match '[^\x00-\x7F]' } | ForEach-Object {$_.Name})
    administrator=$true
    parent_psmodulepath_unchanged=([Environment]::GetEnvironmentVariable('PSModulePath', 'Process') -ceq $originalPSModulePath)
    tracing_enabled=$false
    results=$results
}
[IO.File]::WriteAllText((Join-Path $directory 'result.json'), ($result | ConvertTo-Json -Depth 6), $utf8)
Write-Host "Launch diagnostic complete: $directory"
