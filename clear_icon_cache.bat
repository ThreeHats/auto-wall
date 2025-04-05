@echo off
setlocal enabledelayedexpansion
echo Windows Icon Cache Utility - Safe Version

echo.
echo WARNING: This script will modify the Windows icon cache.
echo The explorer process may need to be restarted during this process.
echo.
echo OPTIONS:
echo 1. Safe mode - Clear icon cache without restarting explorer
echo 2. Full mode - Stop explorer, clear cache, restart explorer (More thorough but may disrupt your desktop)
echo 3. Exit without making changes
echo.

set /p MODE="Choose an option (1, 2, or 3): "

if "%MODE%"=="3" (
    echo Exiting without changes.
    goto :EOF
)

echo.
echo Creating backup of icon cache files...
set BACKUP_DIR=%TEMP%\IconCache_Backup_%RANDOM%
mkdir "%BACKUP_DIR%" 2>nul

if "%MODE%"=="2" (
    echo.
    echo WARNING: Windows Explorer will be temporarily closed.
    echo All windows and desktop icons will disappear briefly.
    echo.
    set /p CONFIRM="Are you sure you want to continue? (Y/N): "
    if /i not "%CONFIRM%"=="Y" goto :EOF
    
    echo.
    echo Stopping Windows Explorer...
    taskkill /f /im explorer.exe
)

echo.
echo Clearing icon cache files...

REM Backup and delete IconCache.db
if exist "%LOCALAPPDATA%\IconCache.db" (
    copy "%LOCALAPPDATA%\IconCache.db" "%BACKUP_DIR%" >nul
    del /a:h "%LOCALAPPDATA%\IconCache.db" 2>nul
)

REM Backup and delete iconcache files
for %%i in ("%LOCALAPPDATA%\iconcache*.*") do (
    copy "%%i" "%BACKUP_DIR%" >nul 2>&1
    del /a:h "%%i" 2>nul
)

REM Clear thumbnail cache safely
if exist "%LOCALAPPDATA%\Microsoft\Windows\Explorer" (
    for %%i in ("%LOCALAPPDATA%\Microsoft\Windows\Explorer\thumbcache*.*") do (
        copy "%%i" "%BACKUP_DIR%" >nul 2>&1
        del "%%i" 2>nul
    )
)

echo.
echo Cache files cleared.
echo Backups saved to: %BACKUP_DIR%

if "%MODE%"=="2" (
    echo.
    echo Restarting Windows Explorer...
    start explorer.exe
)

echo.
echo Process completed!
echo.
echo If your icon still doesn't appear correctly, try:
echo - Right-clicking in File Explorer and selecting "Refresh"
echo - Logging out and back in
echo - Restarting your computer
echo.
pause
