@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set VULKAN_SDK=C:\VulkanSDK\1.4.350.0
cd /d C:\Users\egorb\OneDrive\Documentos\OGSR-Engine
msbuild ogsr_engine\xr_3da\XR_3DA_Vulkan.vcxproj /p:Configuration=Release /p:Platform=x64 /p:SolutionDir=C:\Users\egorb\OneDrive\Documentos\OGSR-Engine\ /m /v:m
echo BUILD_EXIT=%ERRORLEVEL%
