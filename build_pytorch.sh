#!/bin/bash
# build_final.sh

# === 1. 關鍵修正：指向 targets 目錄 ===
# 這是您 find 指令找到的真實路徑
export TARGET_DIR=$CONDA_PREFIX/targets/x86_64-linux

# 設定 CUDA_HOME
export CUDA_HOME=$CONDA_PREFIX

# 將 targets 下的 include 加入編譯器搜尋路徑 (解決 missing CUDA_INCLUDE_DIRS)
export CPATH=${TARGET_DIR}/include:${CONDA_PREFIX}/include:${CPATH}
export C_INCLUDE_PATH=${TARGET_DIR}/include:${CONDA_PREFIX}/include:${C_INCLUDE_PATH}
export CPLUS_INCLUDE_PATH=${TARGET_DIR}/include:${CONDA_PREFIX}/include:${CPLUS_INCLUDE_PATH}

# 將 targets 下的 lib 加入連結器路徑 (解決找不到庫的問題)
export LIBRARY_PATH=${TARGET_DIR}/lib:${CONDA_PREFIX}/lib:${LIBRARY_PATH}
export LD_LIBRARY_PATH=${TARGET_DIR}/lib:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}

# === 2. 基礎 CMake 設定 ===
export CMAKE_PREFIX_PATH=${CONDA_PREFIX}:${CMAKE_PREFIX_PATH}

# === 3. 編譯器與 ABI 設定 ===
export CC=x86_64-conda-linux-gnu-gcc
export CXX=x86_64-conda-linux-gnu-g++
export CMAKE_CXX_COMPILER_LAUNCHER=ccache
export _GLIBCXX_USE_CXX11_ABI=1
export LDFLAGS="-Wl,-rpath,${CONDA_PREFIX}/lib ${LDFLAGS}"

# === 4. NCCL 設定 (強制使用 Conda 版) ===
export USE_SYSTEM_NCCL=1
#export NCCL_ROOT=$CONDA_PREFIX
#export NCCL_INCLUDE_DIR=$CONDA_PREFIX/include
# 注意：有些版本的 NCCL 也會跑去 targets 裡，我們兩邊都設比較保險
export NCCL_ROOT=/home/twyang/nccl/build
export NCCL_INCLUDE_DIR=/home/twyang/nccl/build/include
export NCCL_LIB_DIR=/home/twyang/nccl/build/lib
#export NCCL_LIBRARY=/home/twyang/nccl/build/lib/libnccl.so
export USE_XPU=0
export USE_SYCL=0
# === 5. 功能開關 (A100 優化) ===
export USE_ROCM=0
export USE_CUDA=1
export USE_CUDNN=1
export USE_DISTRIBUTED=1
export TORCH_CUDA_ARCH_LIST="8.0"

# === 6. 執行編譯 ===
echo "Starting PyTorch compilation with FIXED include paths..."
python -m pip install --no-build-isolation -v -e .
