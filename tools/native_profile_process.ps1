# Pure launch helpers; loading this file never starts a process or changes the parent environment.
function New-NativeProfileProcessInfo {
    param([string]$Executable, [string[]]$Arguments, [string]$WorkingDirectory, [switch]$RemovePSModulePath)
    $quoted = foreach ($argument in $Arguments) {
        if ($argument.Contains('"')) { throw 'Unexpected quote in a profiling argument.' }
        '"' + $argument + '"'
    }
    $info = [Diagnostics.ProcessStartInfo]::new()
    $info.FileName = $Executable
    $info.Arguments = ($quoted -join ' ')
    $info.WorkingDirectory = $WorkingDirectory
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.WindowStyle = [Diagnostics.ProcessWindowStyle]::Hidden
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    if ($RemovePSModulePath) { $info.EnvironmentVariables.Remove('PSModulePath') }
    return $info
}

function Test-NativeProfileFailure {
    param([int]$ExitCode, [AllowNull()][string]$StandardError)
    return [bool]($ExitCode -ne 0 -or $StandardError -match 'Unexpected exception|bad conversion|Dynamic exception type')
}

function Initialize-NativeProfileLogWriter {
    if ('JieQiNativeProfile.PipeLog' -as [type]) { return }
    Add-Type -TypeDefinition @'
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;
namespace JieQiNativeProfile {
    public static class PipeLog {
        public static async Task CopyAsync(Stream source, string path) {
            using (var output = new FileStream(path, FileMode.CreateNew, FileAccess.Write,
                                               FileShare.ReadWrite, 4096, true)) {
                var buffer = new byte[8192];
                int count;
                while ((count = await source.ReadAsync(buffer, 0, buffer.Length).ConfigureAwait(false)) != 0) {
                    await output.WriteAsync(buffer, 0, count).ConfigureAwait(false);
                    await output.FlushAsync().ConfigureAwait(false);
                }
            }
        }
        public static void DisposeAfterDrain(Process process, Task stdout, Task stderr) {
            Task.WhenAll(stdout, stderr).ContinueWith(task => {
                // Observe a failed pipe without making cleanup wait for inherited handles.
                var error = task.Exception;
                process.Dispose();
            });
        }
    }
}
'@
}

function Read-NativeProfileLog {
    param([string]$Path)
    $file = [IO.File]::Open($Path, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::ReadWrite)
    $reader = [IO.StreamReader]::new($file, [Text.Encoding]::UTF8, $true)
    try { return $reader.ReadToEnd() } finally { $reader.Dispose() }
}

function Invoke-NativeProfileProcess {
    param([string]$Executable, [string[]]$Arguments, [string]$WorkingDirectory,
        [string]$LogPrefix, [switch]$RemovePSModulePath,
        [ValidateRange(1,3600)][int]$TimeoutSeconds=300,
        [ValidateRange(1,30)][int]$DrainTimeoutSeconds=3)
    foreach ($suffix in @('.stdout.txt', '.stderr.txt', '.process.json')) {
        if (Test-Path -LiteralPath ($LogPrefix + $suffix)) { throw "Refusing to overwrite: $LogPrefix$suffix" }
    }
    Initialize-NativeProfileLogWriter
    $process = [Diagnostics.Process]::new()
    $process.StartInfo = New-NativeProfileProcessInfo -Executable $Executable -Arguments $Arguments `
        -WorkingDirectory $WorkingDirectory -RemovePSModulePath:$RemovePSModulePath
    $stdout = $null
    $stderr = $null
    $utf8 = [Text.UTF8Encoding]::new($false)
    try {
        [void]$process.Start()
        $pidValue = $process.Id
        $state = [ordered]@{pid=$pidValue; started_utc=$process.StartTime.ToUniversalTime().ToString('o')
            executable=$Executable; arguments=$Arguments; timeout_seconds=$TimeoutSeconds
            drain_timeout_seconds=$DrainTimeoutSeconds; exit_code=$null; timed_out=$false
            pipes_complete=$false; failed=$null; processes_terminated=@()}
        [IO.File]::WriteAllText($LogPrefix + '.process.json', ($state | ConvertTo-Json -Depth 5), $utf8)
        $stdout = [JieQiNativeProfile.PipeLog]::CopyAsync($process.StandardOutput.BaseStream, $LogPrefix + '.stdout.txt')
        $stderr = [JieQiNativeProfile.PipeLog]::CopyAsync($process.StandardError.BaseStream, $LogPrefix + '.stderr.txt')
        $timer = [Diagnostics.Stopwatch]::StartNew()
        # The timed overload waits only for the CLI, not pipe EOF held open by orphan workers.
        while (-not $process.WaitForExit(100)) {
            if ($timer.Elapsed.TotalSeconds -ge $TimeoutSeconds) {
                $state.timed_out = $true
                break
            }
            if ($stdout.IsFaulted -or $stderr.IsFaulted) { break }
        }
        if ($process.HasExited) { $state.exit_code = $process.ExitCode }
        $drain = [Diagnostics.Stopwatch]::StartNew()
        while (-not ($stdout.IsCompleted -and $stderr.IsCompleted) `
                -and $drain.Elapsed.TotalSeconds -lt $DrainTimeoutSeconds) { Start-Sleep -Milliseconds 50 }
        $state.pipes_complete = [bool]($stdout.IsCompleted -and $stderr.IsCompleted)
        $state.log_errors = @(foreach ($task in @($stdout,$stderr)) {
            if ($task.IsFaulted) { $task.Exception.ToString() }
        })
        $outText = Read-NativeProfileLog -Path ($LogPrefix + '.stdout.txt')
        $errText = Read-NativeProfileLog -Path ($LogPrefix + '.stderr.txt')
        $state.failed = [bool]($null -eq $state.exit_code -or $state.timed_out -or -not $state.pipes_complete `
            -or $state.log_errors.Count -ne 0 `
            -or (Test-NativeProfileFailure -ExitCode $state.exit_code -StandardError ($outText + "`n" + $errText)))
        $state.finished_utc = (Get-Date).ToUniversalTime().ToString('o')
        [IO.File]::WriteAllText($LogPrefix + '.process.json', ($state | ConvertTo-Json -Depth 5), $utf8)
        return [pscustomobject]@{pid=$pidValue; exit_code=$state.exit_code; stdout=$outText; stderr=$errText
            failed=$state.failed; timed_out=$state.timed_out; pipes_complete=$state.pipes_complete
            log_errors=$state.log_errors}
    } finally {
        if ($null -ne $stdout -and $null -ne $stderr -and -not ($stdout.IsCompleted -and $stderr.IsCompleted)) {
            [JieQiNativeProfile.PipeLog]::DisposeAfterDrain($process, $stdout, $stderr)
        } else { $process.Dispose() }
    }
}
