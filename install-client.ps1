<#
.SYNOPSIS
    Join this Windows machine to the bnk mesh, or remove it.

.DESCRIPTION
    Downloads the latest release for this architecture, installs bnk.exe
    and wintun.dll into C:\Program Files\bnk, registers the bnk service,
    and waits for the tunnel. Re-running it is also the update path.

    Run from an elevated PowerShell:

      & ([scriptblock]::Create((irm https://raw.githubusercontent.com/christ-pher/bnk/main/install-client.ps1))) -Server https://VPS:8443 -Key bnkkey:...

    Uninstall (removes binaries, service, AND this node's identity):

      & ([scriptblock]::Create((irm https://raw.githubusercontent.com/christ-pher/bnk/main/install-client.ps1))) -Uninstall
#>
[CmdletBinding()]
param(
    [string]$Server,
    [string]$Key,
    [switch]$Uninstall
)

$ErrorActionPreference = 'Stop'

$Repo        = 'christ-pher/bnk'
$ServiceName = 'bnk'
$InstallDir  = Join-Path $env:ProgramFiles 'bnk'
$StateDir    = Join-Path $env:ProgramData 'bnk'
$Exe         = Join-Path $InstallDir 'bnk.exe'

function Assert-Admin {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($id)
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw 'Run this from an elevated PowerShell (Run as Administrator).'
    }
}

function Get-Arch {
    switch ($env:PROCESSOR_ARCHITECTURE) {
        'AMD64' { 'amd64' }
        'ARM64' { 'arm64' }
        default { throw "unsupported architecture $env:PROCESSOR_ARCHITECTURE" }
    }
}

# Add-ToMachinePath puts the install dir on the system PATH so `bnk`
# works from any shell, and on this session's PATH so it works right now
# without reopening the terminal.
function Add-ToMachinePath([string]$dir) {
    $machine = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $parts = @($machine -split ';' | Where-Object { $_ -ne '' })
    if ($parts -notcontains $dir) {
        [Environment]::SetEnvironmentVariable('Path', (($parts + $dir) -join ';'), 'Machine')
        Write-Host "added $dir to the system PATH"
    }
    if (@($env:Path -split ';') -notcontains $dir) {
        $env:Path = "$env:Path;$dir"
    }
}

function Remove-FromMachinePath([string]$dir) {
    $machine = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $parts = @($machine -split ';' | Where-Object { $_ -ne '' -and $_ -ne $dir })
    [Environment]::SetEnvironmentVariable('Path', ($parts -join ';'), 'Machine')
}

function Invoke-Uninstall {
    Write-Host 'uninstalling bnk...'
    Remove-FromMachinePath $InstallDir
    if (Get-Service -Name $ServiceName -ErrorAction SilentlyContinue) {
        Stop-Service -Name $ServiceName -Force -ErrorAction SilentlyContinue
        if (Test-Path $Exe) {
            & $Exe service uninstall
        } else {
            sc.exe delete $ServiceName | Out-Null
        }
    }
    foreach ($path in @($InstallDir, $StateDir)) {
        if (Test-Path $path) {
            Write-Host "removing $path"
            Remove-Item -Recurse -Force $path
        }
    }
    Write-Host 'bnk removed. The node identity is gone — rejoining needs a fresh key from the server.'
}

Assert-Admin

if ($Uninstall) {
    Invoke-Uninstall
    return
}

if (-not $Server) {
    throw 'A -Server URL is required, e.g. -Server https://YOUR_VPS:8443'
}

$arch = Get-Arch
$url  = "https://github.com/$Repo/releases/latest/download/bnk-windows-$arch.zip"
$tmp  = Join-Path $env:TEMP ("bnk-" + [guid]::NewGuid())
New-Item -ItemType Directory -Path $tmp -Force | Out-Null

try {
    Write-Host "downloading $url"
    $zip = Join-Path $tmp 'bnk.zip'
    Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing
    Expand-Archive -Path $zip -DestinationPath $tmp -Force

    # Stop the service before replacing files: a running image is locked.
    if (Get-Service -Name $ServiceName -ErrorAction SilentlyContinue) {
        Stop-Service -Name $ServiceName -Force -ErrorAction SilentlyContinue
    }
    New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
    Copy-Item (Join-Path $tmp 'bnk.exe') $Exe -Force
    Copy-Item (Join-Path $tmp 'wintun.dll') (Join-Path $InstallDir 'wintun.dll') -Force
    Copy-Item (Join-Path $tmp 'WINTUN-LICENSE.txt') (Join-Path $InstallDir 'WINTUN-LICENSE.txt') -Force
} finally {
    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}

Add-ToMachinePath $InstallDir

# Show-ServiceDiagnostics reports what the service manager actually
# holds — the registered command line is the thing most likely to be
# wrong, and a bare "cannot start service" never mentions it.
function Show-ServiceDiagnostics {
    Write-Host ''
    Write-Host 'bnk service diagnostics:' -ForegroundColor Yellow
    $svc = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
    Write-Host ("  state         : " + $(if ($svc) { $svc.Status } else { 'not installed' }))
    $wmi = Get-CimInstance -ClassName Win32_Service -Filter "Name='$ServiceName'" -ErrorAction SilentlyContinue
    if ($wmi) {
        Write-Host ("  exit code     : " + $wmi.ExitCode)
        Write-Host ("  command line  : " + $wmi.PathName)
    }
    Write-Host ("  wintun.dll    : " + $(if (Test-Path (Join-Path $InstallDir 'wintun.dll')) { 'present' } else { 'MISSING' }))
    Write-Host ''
    Write-Host 'Run it in the foreground to see the actual error:'
    Write-Host "  & `"$Exe`" run --server $Server --state-dir `"$StateDir`""
}

& $Exe service install --server $Server --key $Key --state-dir $StateDir

# The registered command line must carry the `run` subcommand: a service
# registered as a bare exe starts, prints usage, and exits immediately.
$registered = (Get-CimInstance -ClassName Win32_Service -Filter "Name='$ServiceName'" -ErrorAction SilentlyContinue).PathName
if ($registered -notmatch '\srun(\s|$)') {
    Show-ServiceDiagnostics
    throw "the bnk service was registered without its arguments ($registered) — this build cannot start; please report it."
}

try {
    Start-Service -Name $ServiceName
} catch {
    Show-ServiceDiagnostics
    throw
}

# The first install on a machine also installs the Wintun driver, which
# can take considerably longer than a subsequent start, so this waits
# well past what a warm start needs before giving up.
Write-Host 'waiting for the tunnel (first run also installs the Wintun driver)...'
$joined = $false
foreach ($i in 1..90) {
    $out = & $Exe status 2>$null
    if ($LASTEXITCODE -eq 0) {
        if ($out -match '^bnk is down') {
            & $Exe up | Out-Null      # a past `bnk down` persists on purpose
        } else {
            $joined = $true
            break
        }
    }
    if ($i % 10 -eq 0) { Write-Host "  still waiting ($i s)..." }
    Start-Sleep -Seconds 1
}

if (-not $joined) {
    Show-ServiceDiagnostics
    throw 'bnk did not come up after 90s (see diagnostics above).'
}

# The key is single-use and now spent; the identity lives in $StateDir.
# Re-register without it so the service never resubmits a dead key.
& $Exe service install --server $Server --state-dir $StateDir | Out-Null

Write-Host ''
& $Exe status
Write-Host ''
Write-Host 'Done. Everyday commands (no elevation needed for status):'
Write-Host '    bnk status | bnk ping NAME | bnk netcheck    diagnostics'
Write-Host '    bnk down / bnk up (elevated)                 disconnect / reconnect'
Write-Host ''
Write-Host "bnk is on the system PATH. Open a NEW terminal to pick it up —"
Write-Host 'already-open windows keep their old PATH until restarted.'
