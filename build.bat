@echo off
setlocal enabledelayedexpansion

REM Check if running in automated mode (for CI/CD)
set AUTOMATED=false
if "%1"=="--ci" set AUTOMATED=true
if "%CI%"=="true" set AUTOMATED=true

echo Starting Auto-Wall build process...

REM Verify icon exists and print its path
echo Checking for icon file...
set ICON_PATH=resources\icon.ico
if exist "%ICON_PATH%" (
    echo Icon found at: %CD%\%ICON_PATH%
) else (
    echo WARNING: Icon not found at %CD%\%ICON_PATH%
    echo The executable will be built without an icon.
)

REM Check if PyInstaller is installed
pip show pyinstaller >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Installing PyInstaller...
    pip install pyinstaller --no-cache-dir
    if %ERRORLEVEL% NEQ 0 (
        echo Failed to install PyInstaller. Please install it manually.
        exit /b 1
    )
)

REM Clean PyInstaller cache thoroughly
echo Cleaning PyInstaller cache...
rmdir /s /q build 2>nul
rmdir /s /q __pycache__ 2>nul
rmdir /s /q dist 2>nul
for /d /r %%i in (*__pycache__*) do @rmdir /s /q "%%i" 2>nul

REM Create a clean build environment
echo Creating a clean build environment...
pip install -r requirements.txt --no-cache-dir

REM Run the build using the spec file
echo Running PyInstaller with optimized spec file...
pyinstaller --clean --noconfirm Auto-Wall.spec

if %ERRORLEVEL% NEQ 0 (
    echo Build failed! Check the error messages above.
    if "%AUTOMATED%"=="false" pause
    exit /b 1
)

echo Build complete! Executable is in the dist/Auto-Wall folder.

REM Only show pause in interactive mode
if "%AUTOMATED%"=="false" (
    pause
)

exit /b 0
