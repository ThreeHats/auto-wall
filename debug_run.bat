@echo off
echo Running Auto-Wall in debug mode...
cd dist
Auto-Wall.exe
echo.
echo Application closed with exit code: %ERRORLEVEL%
echo.
pause
