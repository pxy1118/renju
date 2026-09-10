@echo off
rem Keep this console attached to the server so Ctrl+C stops it.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-webui.ps1"
