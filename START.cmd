@echo off
rem ============================================================
rem  QuestEdu - click to run
rem
rem  Starts the server. The API and the web page are served by
rem  the same process, so one click brings up both.
rem
rem  Why this file is long: PATH is not trustworthy. A machine
rem  can have several pythons (IDE-bundled, py launcher, store
rem  app) and the first one on PATH often has none of our
rem  dependencies. So we probe candidates and keep the first one
rem  where "import fastapi, sympy, uvicorn" actually works.
rem
rem  Keep this file ASCII-only: cmd.exe reads .bat/.cmd in the
rem  system codepage, so Chinese text here turns into garbage.
rem  All Chinese output comes from Python.
rem ============================================================
setlocal
chcp 65001 >nul
title QuestEdu
cd /d "%~dp0"
set "PYTHONIOENCODING=utf-8"

set "PYEXE="
set "ANYEXE="

rem --- 1. a project venv wins, no probing needed ---
if exist ".venv\Scripts\python.exe" set PYEXE=".venv\Scripts\python.exe"
if defined PYEXE goto :launch

rem --- 2. py launcher, then whatever "python" happens to be ---
call :probecmd "py -3"
call :probecmd "python"

rem --- 3. python.org installs are not always on PATH; scan them ---
for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python3*") do call :probepath "%%D\python.exe"
for /d %%D in ("%ProgramFiles%\Python3*") do call :probepath "%%D\python.exe"
call :probepath "C:\Python313\python.exe"
call :probepath "C:\Python312\python.exe"
call :probepath "C:\Python311\python.exe"

if defined PYEXE goto :launch

rem --- 4. nothing has the deps: still hand over to *some* python so
rem        the launcher can print the exact pip command ---
if not defined ANYEXE goto :nopython
set PYEXE=%ANYEXE%
goto :launch

:nopython
echo.
echo   [x] Python not found.
echo.
echo   Install Python 3.10 or newer and tick
echo   "Add Python to PATH" during setup:
echo   https://www.python.org/downloads/
echo.
pause
exit /b 1

rem ------------------------------------------------------------
:probecmd
rem bare command such as "py -3"; keep it unquoted
if defined PYEXE exit /b
if not defined ANYEXE set "ANYEXE=%~1"
%~1 -c "import fastapi, sympy, uvicorn" >nul 2>&1
if not errorlevel 1 set "PYEXE=%~1"
exit /b

:probepath
rem real file path; store it quoted so spaces survive
if defined PYEXE exit /b
if not exist "%~1" exit /b
if not defined ANYEXE set ANYEXE="%~1"
"%~1" -c "import fastapi, sympy, uvicorn" >nul 2>&1
if not errorlevel 1 set PYEXE="%~1"
exit /b

rem ------------------------------------------------------------
:launch
echo.
echo   QuestEdu - starting...
echo.

%PYEXE% "scripts\run.py" %*
set "CODE=%ERRORLEVEL%"

if not "%CODE%"=="0" (
    echo.
    echo   [x] Exit code %CODE% - read the message above.
    echo.
    pause
)
endlocal
exit /b %CODE%
