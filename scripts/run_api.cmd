@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_api.ps1" %*
exit /b %ERRORLEVEL%
