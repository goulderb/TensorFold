param([Parameter(Mandatory = $true)][string]$ZigPath)

$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    & $ZigPath test -lc --dep lanes --dep sys '-Mroot=zig/src/core/engine_api.zig' --dep sys '-Mlanes=zig/src/core/lanes/lanes.zig' '-Msys=zig/src/platform/sys.zig' --test-filter 'queue cap follows'
    $queueExit = $LASTEXITCODE
    & $ZigPath test -lc --dep lanes --dep sys '-Mroot=zig/src/core/prompt_cache.zig' --dep sys '-Mlanes=zig/src/core/lanes/lanes.zig' '-Msys=zig/src/platform/sys.zig' --test-filter 'respects pending restore'
    $cacheExit = $LASTEXITCODE
    Write-Output "Queue regression exit: $queueExit; learned-cache regression exit: $cacheExit."
    if ($queueExit -ne 0 -or $cacheExit -ne 0) { exit 1 }
} finally {
    Pop-Location
}
