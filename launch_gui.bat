@echo off
setlocal
cd /d "%~dp0"

set "GUI_PYTHON=%~dp0.venv-gui\Scripts\python.exe"
if not exist "%GUI_PYTHON%" (
    echo The GUI environment has not been installed yet.
    echo Double-click install_gui.bat first.
    echo.
    pause
    exit /b 1
)

echo Starting Lutas Lab Photometry GUI...
echo Keep this window open while using the GUI; you may minimize it.
echo Closing this window will stop the GUI.
echo.
"%GUI_PYTHON%" -m streamlit run "%~dp0streamlit_app.py" --server.address 127.0.0.1 --browser.gatherUsageStats false
set "LAUNCH_RESULT=%ERRORLEVEL%"

if not "%LAUNCH_RESULT%"=="0" (
    echo.
    echo The GUI stopped with exit code %LAUNCH_RESULT%.
    pause
)
exit /b %LAUNCH_RESULT%
