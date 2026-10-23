# Collect tool capabilities only. No recording, Python, training or policy changes.
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = [Security.Principal.WindowsPrincipal]::new($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Open PowerShell with Run as administrator, then run this script again.'
}

$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).ProviderPath
$pilots = (Resolve-Path -LiteralPath (Join-Path $root 'data/pilots')).ProviderPath
$nsys = if ($env:JIEQI_NSYS) { $env:JIEQI_NSYS } else { (Get-Command nsys -ErrorAction Stop).Source }
$wpr = Join-Path $env:SystemRoot 'System32/wpr.exe'
foreach ($executable in @($nsys, $wpr)) {
    if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
        throw "Required installed tool not found: $executable"
    }
    $signature = Get-AuthenticodeSignature -LiteralPath $executable
    if ($signature.Status -ne 'Valid') {
        throw "Cannot verify installed tool signature: $executable ($($signature.Status))"
    }
}

$directory = Join-Path $pilots ('native_profile_preflight_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
if (Test-Path -LiteralPath $directory) {
    throw "Refusing to overwrite existing output: $directory"
}
[void](New-Item -ItemType Directory -Path $directory)

$commands = @(
    @{Name='nsys_version'; Executable=$nsys; Arguments=@('--version')},
    @{Name='nsys_profile_help'; Executable=$nsys; Arguments=@('profile', '--help')},
    @{Name='nsys_stats_help'; Executable=$nsys; Arguments=@('stats', '--help')},
    @{Name='nsys_export_help'; Executable=$nsys; Arguments=@('export', '--help')},
    @{Name='wpr_status'; Executable=$wpr; Arguments=@('-status')}
)
$metadata = [ordered]@{
    administrator=$true
    utc=(Get-Date).ToUniversalTime().ToString('o')
    powershell=$PSVersionTable.PSVersion.ToString()
    workspace=$root
    nsys_sha256=(Get-FileHash -LiteralPath $nsys -Algorithm SHA256).Hash.ToLowerInvariant()
    recording_started=$false
    commands=@()
}
$utf8 = [Text.UTF8Encoding]::new($false)
$resultPath = Join-Path $directory 'result.json'
foreach ($command in $commands) {
    $stdout = Join-Path $directory ($command.Name + '.stdout.txt')
    $stderr = Join-Path $directory ($command.Name + '.stderr.txt')
    $process = Start-Process -FilePath $command.Executable -ArgumentList $command.Arguments `
        -WorkingDirectory $root -WindowStyle Hidden -Wait -PassThru `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    $metadata.commands += [ordered]@{
        name=$command.Name
        exit_code=$process.ExitCode
        stdout_bytes=(Get-Item -LiteralPath $stdout).Length
        stderr_bytes=(Get-Item -LiteralPath $stderr).Length
    }
    [IO.File]::WriteAllText($resultPath, ($metadata | ConvertTo-Json -Depth 5), $utf8)
}
Write-Host "Preflight complete. No recording was started. Output: $directory"
Write-Host 'Keep this administrator terminal open for the later short profiling command.'
