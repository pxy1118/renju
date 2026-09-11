@echo off
rem Same as start-webui.bat, but serves the LAN and prints an invite link for friends.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-webui.ps1" -Share
