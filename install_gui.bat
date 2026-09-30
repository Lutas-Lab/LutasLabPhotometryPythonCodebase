@echo off
setlocal
cd /d "%~dp0"

echo Lutas Lab Photometry GUI installer
echo.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_gui.ps1"
set "INSTALL_RESULT=%ERRORLEVEL%"

echo.
if not "%INSTALL_RESULT%"=="0" (
    echo Installation did not finish successfully.
    echo Review the message above, then try again.
) else (
    echo Installation complete. Double-click launch_gui.bat to start the GUI.
)
echo.
pause
exit /b %INSTALL_RESULT%
