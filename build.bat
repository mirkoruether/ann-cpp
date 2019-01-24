@echo off
set build_tools="C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvarsall.bat"
set myinclude=/I annlib /I linalg

set out_dir=build\debug
set compilerflags=/GL /Gy /Zi /Gm- /sdl /Fd"%out_dir%\vc141.pdb" /Zc:inline /MD /EHsc /nologo /Fo"%out_dir%\obj/"
set linkerflags=/PDB:"%out_dir%/ann-cpp.pdb" /DEBUG /OPT:REF /OPT:ICF
md %out_dir%
md %out_dir%\obj

::call %build_tools% x64
::cl.exe %compilerflags% %myinclude% linalg\*cpp annlib\*.cpp *.cpp /link %linkerflags% /OUT:"%out_dir%\ann-cpp-win32-%VSCMD_ARG_TGT_ARCH%.exe"

set out_dir=build\release
set compilerflags=/GL /Gy /Gm- /Ox /sdl /Zc:inline /Oi /MD /EHsc /nologo /Fo"%out_dir%\obj/"
set linkerflags=/OPT:REF /OPT:ICF
md %out_dir%
md %out_dir%\obj

call %build_tools% x64
cl.exe %compilerflags% %myinclude% linalg\*cpp annlib\*.cpp *.cpp /link %linkerflags% /OUT:"%out_dir%\ann-cpp-win32-%VSCMD_ARG_TGT_ARCH%.exe"

call %build_tools% x64
cl.exe %compilerflags% /openmp %myinclude% linalg\*.cpp annlib\*.cpp *.cpp /link %linkerflags% /OUT:"%out_dir%\ann-cpp-win32-%VSCMD_ARG_TGT_ARCH%-openmp.exe"

::call %build_tools% x86
::cl.exe %compilerflags% /Fo"%out_dir%\obj-x86/" %myinclude% linalg\*cpp annlib\*.cpp *.cpp /link %linkerflags% /OUT:"%out_dir%\ann-cpp--win32-%VSCMD_ARG_TGT_ARCH%.exe"

PAUSE