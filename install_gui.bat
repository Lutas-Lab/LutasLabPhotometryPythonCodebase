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
    echo Press any key to close this window.
) else (
    echo Installation complete.
    echo Press any key to close this window.
    echo Then double-click launch_gui.bat to start the GUI.
)
echo.
pause >nul
exit /b %INSTALL_RESULT%
