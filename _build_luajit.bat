@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
pushd "C:\Users\egorb\OneDrive\Documentos\OGSR-Engine\ogsr_engine\LuaJIT\src"
echo CWD=%CD%
set "NoDefaultCurrentDirectoryInExePath="
set "PATH=%CD%;%PATH%"
call .\msvcbuild.bat x64 static Debug
echo LUAJIT_EXIT=%ERRORLEVEL%
