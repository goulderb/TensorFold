@echo off
rem nvcc wrapper for the Windows TensorFold build.
rem
rem CUDA 12.8's host_config.h refuses any host compiler newer than VS 2022
rem (MSVC 19.4x). When both VS 2022 and a newer Visual Studio are installed,
rem nvcc auto-detects the newest one and fails with C1189. CUDAHOSTCXX does
rem not override that check; only -ccbin does. This wrapper pins nvcc to the
rem newest MSVC toolset inside a VS 2022 install.
rem
rem Pass it to the build as: -Dnvcc=<this file>
setlocal

set "PF86=%ProgramFiles(x86)%"
set "PF=%ProgramFiles%"
set "CCBIN="

for %%r in (
  "%PF86%\Microsoft Visual Studio\2022\BuildTools"
  "%PF86%\Microsoft Visual Studio\2022\Community"
  "%PF86%\Microsoft Visual Studio\2022\Professional"
  "%PF86%\Microsoft Visual Studio\2022\Enterprise"
  "%PF%\Microsoft Visual Studio\2022\BuildTools"
  "%PF%\Microsoft Visual Studio\2022\Community"
  "%PF%\Microsoft Visual Studio\2022\Professional"
  "%PF%\Microsoft Visual Studio\2022\Enterprise"
) do (
  if not defined CCBIN if exist "%%~r\VC\Tools\MSVC" (
    for /f "delims=" %%d in ('dir /b /ad /o-n "%%~r\VC\Tools\MSVC" 2^>nul') do (
      if not defined CCBIN if exist "%%~r\VC\Tools\MSVC\%%d\bin\Hostx64\x64\cl.exe" set "CCBIN=%%~r\VC\Tools\MSVC\%%d\bin\Hostx64\x64"
    )
  )
)

if not defined CCBIN (
  echo nvcc-msvc2022: no VS 2022 x64 MSVC toolset found 1>&2
  exit /b 1
)

set "NVCC=%CUDA_PATH%\bin\nvcc.exe"
if not exist "%NVCC%" set "NVCC=%PF%\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe"
if not exist "%NVCC%" (
  echo nvcc-msvc2022: nvcc.exe not found 1>&2
  exit /b 1
)

"%NVCC%" -ccbin "%CCBIN%" %*
exit /b %ERRORLEVEL%
