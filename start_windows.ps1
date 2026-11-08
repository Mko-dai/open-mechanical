# open-mechanical を Windows で起動する（start.bat から呼ばれます）
# 初回は Python と必要なライブラリを、このフォルダの中（.tools と .venv）に自動で入れます。
# パソコン全体には何もインストールしません（このフォルダを消せば元どおり）。
$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$Tools = Join-Path $Root ".tools"
$Uv = Join-Path $Tools "uv\uv.exe"
$Py = Join-Path $Root ".venv\Scripts\python.exe"
$PyW = Join-Path $Root ".venv\Scripts\pythonw.exe"
# uv が取ってくる Python とキャッシュもこのフォルダの中に置く
$env:UV_PYTHON_INSTALL_DIR = Join-Path $Tools "python"
$env:UV_CACHE_DIR = Join-Path $Tools "cache"

function Fail($msg) {
    Write-Host ""
    Write-Host "エラー: $msg" -ForegroundColor Red
    Write-Host "このウィンドウを閉じるには Enter を押してください。"
    [void](Read-Host)
    exit 1
}

# ---- GitHub と同期（git が入っていて、git clone したフォルダのときだけ）
if ((Get-Command git -ErrorAction SilentlyContinue) -and (Test-Path (Join-Path $Root ".git"))) {
    Write-Host "=== GitHub と同期しています ==="
    try {
        git fetch -q origin 2>$null
        git diff --quiet HEAD 2>$null
        if ($LASTEXITCODE -eq 0) {
            git merge -q --ff-only origin/main 2>$null
            if ($LASTEXITCODE -ne 0) { git reset -q --keep origin/main 2>$null }
        } else {
            Write-Host "（手元で変更されたファイルがあるため自動更新はスキップしました）"
        }
    } catch { Write-Host "（同期できませんでした。今のファイルのまま起動します）" }
}

# ---- uv（Python とライブラリを入れる道具）を取ってくる
if (-not (Test-Path $Uv)) {
    $arch = $env:PROCESSOR_ARCHITECTURE
    if ($arch -eq "ARM64") {
        Fail "ARM 版の Windows には、必要なライブラリ（gmsh）の配布がないため対応していません。"
    }
    Write-Host "=== 初回セットアップ（10 分ほどかかります。インターネット接続が必要です） ==="
    Write-Host "uv をダウンロードしています..."
    New-Item -ItemType Directory -Force -Path (Join-Path $Tools "uv") | Out-Null
    $zip = Join-Path $Tools "uv.zip"
    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -UseBasicParsing -Uri "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip" -OutFile $zip -ErrorAction Stop
        Expand-Archive -Force -Path $zip -DestinationPath (Join-Path $Tools "uv") -ErrorAction Stop
        Remove-Item $zip
    } catch { Fail "uv のダウンロードに失敗しました（$($_.Exception.Message)）。インターネット接続を確認してください。" }
    if (-not (Test-Path $Uv)) {
        $found = Get-ChildItem -Recurse -Filter uv.exe (Join-Path $Tools "uv") | Select-Object -First 1
        if ($found) { Copy-Item $found.FullName $Uv } else { Fail "uv.exe が見つかりません。" }
    }
}

# ---- Python（3.12）と仮想環境
if (-not (Test-Path $Py)) {
    Write-Host "Python 3.12 を用意しています..."
    & $Uv venv --python 3.12 (Join-Path $Root ".venv")
    if ($LASTEXITCODE -ne 0) { Fail "Python の準備に失敗しました。" }
}

# ---- ライブラリ（requirements が変わったときだけ入れ直す）
$reqText = (Get-Content -Raw requirements.txt) + (Get-Content -Raw requirements-fast.txt -ErrorAction SilentlyContinue)
$sha = [System.Security.Cryptography.SHA256]::Create()
$hash = ([BitConverter]::ToString($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($reqText))) -replace "-", "").Substring(0, 12)
$hashFile = Join-Path $Root ".venv\.req_hash"
if (-not (Test-Path $hashFile) -or ((Get-Content $hashFile -ErrorAction SilentlyContinue) -ne $hash)) {
    Write-Host "=== ライブラリをインストールしています（初回は数分かかります） ==="
    & $Uv pip install --python $Py -r requirements.txt
    if ($LASTEXITCODE -ne 0) { Fail "ライブラリのインストールに失敗しました。" }
    # 高速化用（入らなくても動作します）
    foreach ($pkg in @((Get-Content requirements-fast.txt -ErrorAction SilentlyContinue) + "pypardiso")) {
        $p = "$pkg".Trim()
        if ($p -eq "" -or $p.StartsWith("#")) { continue }
        & $Uv pip install --python $Py $p 2>$null
        if ($LASTEXITCODE -ne 0) { Write-Host "（$p はインストールできませんでした。標準のソルバーを使います）" }
    }
    Set-Content -Path $hashFile -Value $hash
}

# ---- 起動（黒いウィンドウは閉じて、ソフトだけを開く）
Write-Host "=== open-mechanical を起動します ==="
$exe = if (Test-Path $PyW) { $PyW } else { $Py }
Start-Process -FilePath $exe -ArgumentList "-m", "openmech" -WorkingDirectory $Root
