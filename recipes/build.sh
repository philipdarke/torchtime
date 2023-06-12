#!/bin/bash

set -euxo pipefail

if [[ ${cuda_compiler_version} != "None" && "$target_platform" == linux-64 ]]; then
    export TORCH_CUDA_ARCH_LIST="3.5;5.0"
    export FORCE_CUDA="1"
    export CC="$GCC"
    if [[ ${cuda_compiler_version} == 9.0* ]]; then
        export TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST;6.0;7.0+PTX"
    elif [[ ${cuda_compiler_version} == 9.2* ]]; then
        export TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST;6.0;6.1;7.0+PTX"
    elif [[ ${cuda_compiler_version} == 10.* ]]; then
        export TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST;6.0;6.1;7.0;7.5+PTX"
    elif [[ ${cuda_compiler_version} == 11.0* ]]; then
        export TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST;6.0;6.1;7.0;7.5;8.0+PTX"
    elif [[ ${cuda_compiler_version} == 11.1 ]]; then
        export TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST;6.0;6.1;7.0;7.5;8.0;8.6+PTX"
    elif [[ ${cuda_compiler_version} == 11.2 ]]; then
        export TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST;6.0;6.1;7.0;7.5;8.0;8.6+PTX"
    else
        echo "unsupported cuda version. edit build_pytorch.sh"
        exit 1
    fi
fi


echo "Installing"
${PYTHON} -m pip install . -vv
