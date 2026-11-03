@echo off
rem Starts Cinder's UCI engine in WSL, for a chess GUI on Windows such as Cute Chess or Arena:
rem add this file to the GUI as a UCI engine. Set the WSL distro, the repository in it, and the
rem run and generation to play. A GPU needs a minute to compile kernels before the first readyok.
set DISTRO=archlinux
set REPO=/home/kevin/cinder-chess-zero
set NETWORK=runs/long 90
wsl.exe -d %DISTRO% --cd %REPO% -- env CUDA_PATH=/opt/cuda ./target/release/cinder uci %NETWORK% --backend cuda-bf16
