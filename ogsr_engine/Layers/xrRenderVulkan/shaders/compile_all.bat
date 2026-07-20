@echo off
REM ===========================================================================
REM Compile ALL Vulkan stage shaders (vert/frag/comp .glsl) in this directory
REM to SPIR-V (.spv) next to the source.
REM
REM Shared includes (light_ubo.glsl, shadow_common.glsl, wet_common.glsl,
REM froxel.glsl, ...) are pulled in automatically by glslc via #include
REM (GL_GOOGLE_include_directive). They have no stage suffix so the patterns
REM below skip them - they are not compiled standalone.
REM
REM Usage:  compile_all.bat            compile, then deploy if the game dir exists
REM         compile_all.bat nodeploy   compile only
REM ===========================================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"

REM Locate glslc: prefer the VULKAN_SDK env var, else the known install here.
if not defined VULKAN_SDK set "VULKAN_SDK=C:\VulkanSDK\1.4.350.0"
set "GLSLC=%VULKAN_SDK%\Bin\glslc.exe"
if not exist "%GLSLC%" (
    echo ERROR: glslc.exe not found at "%GLSLC%"
    echo Set the VULKAN_SDK environment variable to your Vulkan SDK install.
    exit /b 1
)
echo Using %GLSLC%
echo.

set ERR=0
for %%f in (*.vert.glsl) do call :one "%%f" vertex
for %%f in (*.frag.glsl) do call :one "%%f" fragment
for %%f in (*.comp.glsl) do call :one "%%f" compute
for %%f in (*.tesc.glsl) do call :one "%%f" tesscontrol
for %%f in (*.tese.glsl) do call :one "%%f" tesseval

if %ERR% NEQ 0 (
    echo.
    echo BUILD FAILED: %ERR% shader had errors.
    exit /b 1
)
echo.
echo All shaders compiled OK.

REM Optional deploy to the test game install (see memory: vulkan-build-deploy).
if /I "%~1"=="nodeploy" goto :done
set "DEPLOY=D:\OGSR Gunslinger Addon\gamedata\shaders"
if exist "%DEPLOY%\" (
    echo Deploying *.spv to "%DEPLOY%" ...
    copy /Y *.spv "%DEPLOY%\" >nul
    echo Deploy complete.
) else (
    echo Deploy dir "%DEPLOY%" not found - skipping deploy.
)

:done
exit /b 0

:one
REM %1 = source .glsl (quoted), %2 = shader stage; output = name-without-.glsl + .spv
REM --target-env=vulkan1.3: the engine requires a VK 1.3 device (HWCaps_Vulkan.h),
REM and vsm_mark.comp subgroup ops need SPIR-V >= 1.3 (default target is vulkan1.0).
"%GLSLC%" --target-env=vulkan1.3 -fshader-stage=%2 "%~1" -o "%~n1.spv"
if errorlevel 1 (
    echo   FAILED: %~1
    set /a ERR+=1
) else (
    echo   ok: %~1
)
goto :eof
