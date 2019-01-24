@echo off
set build_tools="C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvarsall.bat"
set myinclude=/I annlib /I linalg

set out_dir=build\release
::Error: Net not converging when /Zi is not set
set compilerflags=/EHsc /W3 /nologo /Ox /Oi /MD /GL /Gy /Zc:inline /Zi
set linkerflags=/OPT:REF /OPT:ICF

set nvcc="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin\nvcc.exe"
set cuda_options=-link -ccbin cl.exe -Xcompiler="%compilerflags%"
set cuda_include=-Ilinalg -Ilinalg\cuda -Iannlib -Iannlib\cuda
set defs=-D_WINDOWS -DLINALG_CUDA_SUPPORT -DANNLIB_USE_CUDA -DNDEBUG

setlocal EnableDelayedExpansion

set cpp_sources=
set cu_sources=

for %%g in (linalg\*.cpp) do set cpp_sources=!cpp_sources! %%g
for %%g in (annlib\*.cpp) do set cpp_sources=!cpp_sources! %%g
for %%g in (*.cpp) do set cpp_sources=!cpp_sources! %%g

for %%g in (linalg\cuda\*.cu) do set cu_sources=!cu_sources! %%g
for %%g in (annlib\cuda\*.cu) do set cu_sources=!cu_sources! %%g

echo %cpp_sources%
echo %cu_sources%

md %out_dir%

call %build_tools% x64
call %nvcc% %cuda_options% %cuda_include% %defs% -o "%out_dir%\ann-cpp-win32-%VSCMD_ARG_TGT_ARCH%-cuda.exe" %cu_sources% %cpp_sources%

PAUSE