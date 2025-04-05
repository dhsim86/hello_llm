@echo off

:: 1. Visual Studio Compiler Path Setting
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

:: 2. Python 경로 설정
set PYTHON_ROOT=E:\workspace\python\Python-3.11.9

set INCLUDE=%INCLUDE%;%PYTHON_ROOT%\Include;%PYTHON_ROOT%\PC
set LIB=%LIB%;%PYTHON_ROOT%\Libs;%PYTHON_ROOT%\PCbuild\amd64

echo [INFO] Python Header and Library Path Setting Completed
cmd
