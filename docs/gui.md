# Streamlit interface

The Streamlit app provides a browser-based session-manifest editor and
launchers for batch preprocessing and event-aligned PSTHs. It calls the same
maintained command-line workflows as PowerShell and Jupyter.

## Windows installer

For a lab computer without an existing project environment:

1. Install 64-bit Python 3.12 and enable **Add python.exe to PATH**.
2. Download or clone this repository.
3. Double-click `install_gui.bat` once.
4. Double-click `launch_gui.bat` to start the interface.

The installer creates `.venv` inside the repository and does not modify other
Python environments. Internet access is required for the initial installation.
If the repository is moved or renamed, run `install_gui.bat` again. Do not copy
`.venv` between computers.

`install_gui.ps1` contains the PowerShell installation logic; the batch file is
the double-clickable wrapper.

## Launch from the uv environment

Users of the standard project environment can install the GUI extra and launch
the app directly:

```powershell
uv sync --frozen --all-packages --extra gui
uv run python -m streamlit run streamlit_app.py
```

The interface opens locally in a browser. **Preview analysis commands only** is
enabled by default: analysis buttons display the exact command without running
it, while manifest validation, saving, and CSV downloads remain available.
Disable preview only after checking the manifest, data root, and output path.
Existing processed files remain protected unless overwrite is explicitly
enabled.
