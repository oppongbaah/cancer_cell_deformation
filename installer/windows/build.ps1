<#
  Build the celldform Windows installer.   Run on Windows, from anywhere:   powershell -File installer\windows\build.ps1

  Needs: Python 3.10-3.12 on PATH and Inno Setup 7 or 6 (iscc.exe). Output: dist\installer\celldform-setup-<version>.exe + .sha256
  Options: -Gpu        bundle the CUDA build of PyTorch (about 2.5 GB bigger) instead of the CPU build
           -Python X   which python to use          -SkipInstaller   stop after the program folder (dist\celldform)

  The build runs in a NEUTRAL folder (C:\celldform_build): compiled Python files remember the folder they were built
  from, and that must not be the maker's own home folder. Then installer\audit_package.py scans what is about to ship
  and the build stops if it finds the maker's e-mail, user name or home folder, or a data or secret file.
#>
param([switch]$Gpu, [string]$Python = "python", [switch]$SkipInstaller)
$ErrorActionPreference = "Stop"
function Step($m) { Write-Host "`n== $m" -ForegroundColor Cyan }
function Run($exe, $argList) { & $exe @argList; if ($LASTEXITCODE -ne 0) { throw "$exe failed ($LASTEXITCODE)" } }

$repo  = (Resolve-Path "$PSScriptRoot\..\..").Path
$stage = Join-Path $env:SystemDrive "celldform_build"
$src   = Join-Path $stage "src"
$venv  = Join-Path $stage "venv"
$vpy   = Join-Path $venv "Scripts\python.exe"

Step "Version"
$version = (& $Python -c "import tomllib;print(tomllib.load(open(r'$repo\pyproject.toml','rb'))['project']['version'])").Trim()
Write-Host "celldform $version"

Step "Staging a clean copy in $stage (no compiled files, no data, no git history)"
if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
New-Item -ItemType Directory -Path $src | Out-Null
foreach ($d in "celldform", "installer") {
  robocopy "$repo\$d" "$src\$d" /E /XD __pycache__ /XF *.pyc /NFL /NDL /NJH /NJS | Out-Null
  if ($LASTEXITCODE -ge 8) { throw "robocopy failed" }
}
Copy-Item "$repo\LICENSE" $src
Copy-Item "$repo\pyproject.toml" $src

Step "Audit of the source that will be built (our own files must carry nothing personal)"
Run $Python @("$repo\installer\audit_package.py", "$src\celldform", "--own", "celldform")

Step "Python environment"
Run $Python @("-m", "venv", $venv)
Run $vpy @("-m", "pip", "install", "--upgrade", "pip", "--quiet")
if ($Gpu) { Run $vpy @("-m", "pip", "install", "torch", "--quiet") }
else      { Run $vpy @("-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/cpu", "--quiet") }
Run $vpy @("-m", "pip", "install", "-r", "$src\installer\windows\requirements-app.txt", "pyinstaller", "--quiet")

Step "Third-party licence notices"
& $vpy "$src\installer\third_party_notices.py" | Out-File -Encoding utf8 "$src\THIRD_PARTY_NOTICES.txt"

Step "Freezing the program (PyInstaller)"
Push-Location $stage
try {
  Run $vpy @("-m", "PyInstaller", "--noconfirm", "--clean", "--distpath", "$stage\dist", "--workpath", "$stage\work",
             "$src\installer\windows\celldform.spec")
} finally { Pop-Location }

Step "Does the finished program start? (--selftest builds the window without a screen, models or files)"
$env:QT_QPA_PLATFORM = "offscreen"
Run "$stage\dist\celldform\celldform.exe" @("--selftest")
Remove-Item Env:QT_QPA_PLATFORM

Step "Audit of everything that will be shipped (the maker's identifiers, data and secret files)"
Run $vpy @("$src\installer\audit_package.py", "$stage\dist\celldform", "--own", "celldform")

New-Item -ItemType Directory -Force -Path "$repo\dist" | Out-Null
if (Test-Path "$repo\dist\celldform") { Remove-Item "$repo\dist\celldform" -Recurse -Force }
Copy-Item "$stage\dist\celldform" "$repo\dist\celldform" -Recurse
if ($SkipInstaller) { Write-Host "`nProgram folder: $repo\dist\celldform"; exit 0 }

Step "Building the installer (Inno Setup)"
$iscc = @("7", "6") | ForEach-Object { $v = $_
          "$env:ProgramFiles\Inno Setup $v\ISCC.exe", "$env:ProgramFiles(x86)\Inno Setup $v\ISCC.exe",
          "$env:LOCALAPPDATA\Programs\Inno Setup $v\ISCC.exe" } |
        Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $iscc) { $iscc = (Get-Command iscc -ErrorAction SilentlyContinue).Source }
if (-not $iscc) { throw "Inno Setup was not found (version 7 or 6). Install it from https://jrsoftware.org/isdl.php, or: winget install --id JRSoftware.InnoSetup.7 -e" }
Run $iscc @("/DAppVersion=$version", "$repo\installer\windows\celldform.iss")

$exe = "$repo\dist\installer\celldform-setup-$version.exe"
$hash = (Get-FileHash $exe -Algorithm SHA256).Hash.ToLower()
"$hash  $(Split-Path $exe -Leaf)" | Out-File -Encoding ascii "$exe.sha256"

Step "Audit of the installer file itself"
Run $vpy @("$src\installer\audit_package.py", $exe, "--own", "celldform")

Write-Host "`nDone." -ForegroundColor Green
Write-Host "  $exe"
Write-Host "  SHA-256 $hash   (give this to whoever installs it so they can check the download)"
Write-Host "  The installer is NOT code-signed: Windows SmartScreen will warn on first run. See installer\README.md."
