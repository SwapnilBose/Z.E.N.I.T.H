@echo off
setlocal
cd /d "%~dp0"

REM Default MSYS2 UCRT64 runtime path. Edit this if MSYS2 is installed elsewhere.
if exist "C:\msys64\ucrt64\bin" set "PATH=C:\msys64\ucrt64\bin;%PATH%"

REM Common eSpeak NG installer locations.
if exist "%ProgramFiles%\eSpeak NG" set "PATH=%ProgramFiles%\eSpeak NG;%PATH%"
if exist "%ProgramFiles(x86)%\eSpeak NG" set "PATH=%ProgramFiles(x86)%\eSpeak NG;%PATH%"

if not exist "build\zenith.exe" (
  echo Z.E.N.I.T.H. has not been built yet. See README.md.
  pause
  exit /b 1
)

build\zenith.exe %*
if errorlevel 1 (
  echo.
  echo Z.E.N.I.T.H. exited with an error.
  pause
)
endlocal
