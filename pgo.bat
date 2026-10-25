@echo off
setlocal

rem Build a profile-guided optimized engine.
rem Works from any directory: every path is resolved relative to this script.
rem Pass any argument (e.g. "pgo.bat ci") to skip the final pause.

cd /d "%~dp0"

echo [1/3] Building the instrumented engine...
cargo pgo instrument
if errorlevel 1 goto error

echo.
echo [2/3] Running the benchmark to collect profile data...
cargo pgo run -- bench
if errorlevel 1 goto error

echo.
echo [3/3] Building the optimized engine...
cargo pgo optimize
if errorlevel 1 goto error

rem cargo-pgo writes into target\<host triple>\release, while plain
rem "cargo build --release" and most chess GUIs expect target\release.
if not exist "target\release" mkdir "target\release"

set "PGO_BIN="
for %%T in (x86_64-pc-windows-msvc x86_64-pc-windows-gnu aarch64-pc-windows-msvc) do (
    if not defined PGO_BIN if exist "target\%%T\release\Ruthless.exe" (
        copy /Y "target\%%T\release\Ruthless.exe" "target\release\Ruthless.exe" >nul
        if not errorlevel 1 (
            copy /Y "target\release\Ruthless.exe" "target\release\Ruthless-pgo.exe" >nul
            set "PGO_BIN=target\%%T\release\Ruthless.exe"
        )
    )
)

if not defined PGO_BIN goto notfound

echo.
echo Build complete.
echo Optimized binary: target\release\Ruthless.exe
echo Copy kept as:    target\release\Ruthless-pgo.exe
echo Copied from: %PGO_BIN%
if "%~1"=="" pause
exit /b 0

:notfound
echo.
echo Warning: the optimized binary was not found under target\<triple>\release.
echo Look for Ruthless.exe in the paths printed by "cargo pgo optimize".
goto error

:error
echo.
echo Build failed. Check the output above for details.
if "%~1"=="" pause
exit /b 1
