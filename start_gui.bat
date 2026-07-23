@echo off
cd /d "%~dp0"

:: === 检测 Python 解释器：优先 py -3，其次 python ===
set "PY_CMD="
py -3 --version >nul 2>&1
if not errorlevel 1 (
    set "PY_CMD=py -3"
) else (
    python --version >nul 2>&1
    if not errorlevel 1 set "PY_CMD=python"
)

:: === 检查 --dry-run 参数 ===
echo %* | findstr /i /c:"--dry-run" >nul
if not errorlevel 1 (
    if "%PY_CMD%"=="" (
        echo [错误] 未找到 Python。请先安装 Python 3.x。
        pause
        exit /b 1
    )
    echo [dry-run] 检测到 Python：%PY_CMD%
    echo [dry-run] 将执行的命令：%PY_CMD% entry_gui.py %*
    exit /b 0
)

:: === 启动 GUI ===
if "%PY_CMD%"=="" (
    echo [错误] 未找到 Python。请先安装 Python 3.x。
    pause
    exit /b 1
)

%PY_CMD% entry_gui.py
if errorlevel 1 (
    echo.
    echo [提示] 启动返回非零状态码。请尝试先执行：
    echo   pip install -r "%~dp0requirements.txt"
    pause
)