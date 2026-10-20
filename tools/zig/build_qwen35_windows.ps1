#requires -Version 7.0
[CmdletBinding()]
param(
    [ValidateSet('test-qwen35-cuda-host', 'test-qwen35-sampler-turn', 'native', 'fatbins', 'qwen35-probe', 'qwen35-kernels-probe', 'qwen35-head-cost', 'qwen35-sampler-ab', 'qwen35-decode-profile')]
    [string]$Target = 'test-qwen35-cuda-host',
    [Parameter(Mandatory = $true)]
    [string]$Zig,
    [string]$CudaRoot = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8',
    [string]$BuildTools = 'C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools',
    [string]$MsvcVersion = '14.44.35207',
    [ValidateRange(1, 32)]
    [int]$Jobs = 2,
    [string]$SourceCommit = '',
    [string]$PrebuiltFatbins = '',
    [switch]$WithoutKernels
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
if (-not $SourceCommit) {
    $SourceCommit = (& git -C $repo rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0) { throw 'Cannot resolve probe source revision' }
}
if ($SourceCommit -cnotmatch '^[0-9a-f]{40}$') { throw 'SourceCommit must be a full lowercase commit SHA' }
$vcvars = Join-Path $BuildTools 'VC/Auxiliary/Build/vcvars64.bat'
$compiler = Join-Path $BuildTools "VC/Tools/MSVC/$MsvcVersion/bin/Hostx64/x64/cl.exe"
$nvcc = Join-Path $CudaRoot 'bin/nvcc.exe'
$requiredTools = @($Zig, $vcvars, $compiler)
if ($PrebuiltFatbins) {
    if ($WithoutKernels) { throw 'PrebuiltFatbins and WithoutKernels cannot be combined' }
    $PrebuiltFatbins = (Resolve-Path -LiteralPath $PrebuiltFatbins).Path
    if (-not (Test-Path -LiteralPath $PrebuiltFatbins -PathType Container)) {
        throw 'PrebuiltFatbins must be a directory'
    }
} elseif (-not $WithoutKernels) {
    $requiredTools += $nvcc
}
foreach ($path in $requiredTools) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required build tool is missing: $path"
    }
}

$vcInfo = [System.Diagnostics.ProcessStartInfo]::new('cmd.exe')
$vcInfo.Arguments = "/d /c call `"$vcvars`" -vcvars_ver=$MsvcVersion >nul && set"
$vcInfo.UseShellExecute = $false
$vcInfo.RedirectStandardOutput = $true
$vcInfo.RedirectStandardError = $true
$vcProcess = [System.Diagnostics.Process]::Start($vcInfo)
$vcOutput = $vcProcess.StandardOutput.ReadToEndAsync()
$vcErrors = $vcProcess.StandardError.ReadToEndAsync()
$vcProcess.WaitForExit()
$vcText = $vcOutput.GetAwaiter().GetResult()
$vcErrorText = $vcErrors.GetAwaiter().GetResult()
if ($vcProcess.ExitCode -ne 0) {
    throw "MSVC environment setup failed with exit $($vcProcess.ExitCode): $vcErrorText"
}

$build = [System.Diagnostics.ProcessStartInfo]::new($Zig)
$build.UseShellExecute = $false
$build.RedirectStandardOutput = $true
$build.RedirectStandardError = $true
$build.WorkingDirectory = $repo
foreach ($line in ($vcText -split '\r?\n')) {
    if ($line -match '^([^=]+)=(.*)$') {
        $build.Environment[$Matches[1]] = $Matches[2]
    }
}
if ($build.Environment['VCToolsVersion'].TrimEnd('\') -ne $MsvcVersion) {
    throw "vcvars selected unexpected toolset $($build.Environment['VCToolsVersion'])"
}
$build.Environment['PATH'] = (Split-Path -Parent $compiler) + ';' + $build.Environment['PATH']
$build.Environment['INCLUDE'] = (Join-Path $BuildTools "VC/Tools/MSVC/$MsvcVersion/include") + ';' + $build.Environment['INCLUDE']
foreach ($argument in @('build', $Target, "-j$Jobs", '--summary', 'all')) {
    $build.ArgumentList.Add($argument)
}
$build.ArgumentList.Add("-Dqwen35-probe-source=$SourceCommit")
if ($PrebuiltFatbins) {
    $build.ArgumentList.Add("-Dfatbins=$PrebuiltFatbins")
} elseif (-not $WithoutKernels) {
    $build.ArgumentList.Add("-Dnvcc=$nvcc")
    $build.ArgumentList.Add('-Dsm=120')
}
Write-Host "MSVC $MsvcVersion; CUDA 12.8; target $Target; kernels $(-not $WithoutKernels)"
$process = [System.Diagnostics.Process]::Start($build)
$buildOutput = $process.StandardOutput.ReadToEndAsync()
$buildErrors = $process.StandardError.ReadToEndAsync()
$process.WaitForExit()
$outputText = $buildOutput.GetAwaiter().GetResult()
$errorText = $buildErrors.GetAwaiter().GetResult()
if ($outputText.Length -gt 0) { Write-Output $outputText.TrimEnd() }
if ($errorText.Length -gt 0) { Write-Output $errorText.TrimEnd() }
exit $process.ExitCode
