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

function Invoke-Uninstall {
    Write-Host 'uninstalling bnk...'
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

& $Exe service install --server $Server --key $Key --state-dir $StateDir
Start-Service -Name $ServiceName

Write-Host 'waiting for the tunnel...'
$joined = $false
foreach ($i in 1..30) {
    $out = & $Exe status 2>$null
    if ($LASTEXITCODE -eq 0) {
        if ($out -match '^bnk is down') {
            & $Exe up | Out-Null      # a past `bnk down` persists on purpose
        } else {
            $joined = $true
            break
        }
    }
    Start-Sleep -Seconds 1
}
if (-not $joined) {
    throw "bnk did not come up after 30s. Check: Get-EventLog -LogName Application -Source bnk, or run `"$Exe`" run --server $Server by hand."
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
Write-Host "Add $InstallDir to your PATH to run bnk from anywhere."
