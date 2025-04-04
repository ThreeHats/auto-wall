@echo off
echo Starting Auto-Wall build process...

REM Check if PyInstaller is installed
pip show pyinstaller >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Installing PyInstaller...
    pip install pyinstaller
    if %ERRORLEVEL% NEQ 0 (
        echo Failed to install PyInstaller. Please install it manually.
        exit /b 1
    )
)

REM Clean PyInstaller cache more thoroughly
echo Cleaning PyInstaller cache...
rmdir /s /q build
rmdir /s /q __pycache__
rmdir /s /q dist
for /d /r %%i in (*__pycache__*) do @rmdir /s /q "%%i"

REM Create a clean build environment
echo Creating a clean build environment...
pip install -r requirements.txt

REM Make sure scikit-learn and NumPy are properly installed with compatible versions
echo Ensuring scikit-learn and NumPy are properly installed...
pip uninstall -y numpy scikit-learn
pip install numpy==1.26.4 scikit-learn==1.6.1

REM Run the build using the spec file
echo Running PyInstaller with optimized spec file...
pyinstaller --clean Auto-Wall.spec

if %ERRORLEVEL% NEQ 0 (
    echo Build failed! Check the error messages above.
    pause
    exit /b 1
)

echo Build complete! Executable is in the dist/Auto-Wall folder.

REM Copy the debug launcher to the dist folder
echo Creating debug launcher...
copy debug_run.bat dist\debug_run.bat

echo.
echo To test the application in debug mode, run debug_run.bat
echo.
pause
