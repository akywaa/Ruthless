@echo off

echo Building the instrumented engine...
cargo pgo instrument
if %ERRORLEVEL% NEQ 0 goto error

echo.
echo Running the benchmark to collect profiling data...
cargo pgo run -- bench
if %ERRORLEVEL% NEQ 0 goto error

echo.
echo Building the optimized engine...
cargo pgo optimize
if %ERRORLEVEL% NEQ 0 goto error

echo.
echo Copying the optimized binary to targetrelease...
copy Y targetx86_64-pc-windows-msvcreleaseRuthless.exe targetreleaseRuthless.exe 2nul
if %ERRORLEVEL% NEQ 0 (
    copy Y targetx86_64-pc-windows-gnureleaseRuthless.exe targetreleaseRuthless.exe 2nul
)

echo.
echo Build complete. The optimized binary should be in targetrelease.
goto end

error
echo.
echo Build failed. Check the output above for details.

end
pause