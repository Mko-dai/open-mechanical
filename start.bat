@echo off
rem open-mechanical starter for Windows (double-click)
rem First run installs Python and libraries into this folder automatically.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_windows.ps1"
if errorlevel 1 pause
