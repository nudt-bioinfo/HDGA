#!/bin/bash
set -euo pipefail

cd -- "$(dirname -- "$0")"

# Select build type
echo "Select build type:"
echo "1) CPU"
echo "2) GPU"
echo "3) Hybrid (CPU+GPU scheduling)"
read -p "Enter choice (1, 2, or 3): " choice

case $choice in
    1)
        build_type="cpu"
        ;;
    2)
        build_type="gpu"
        ;;
    3)
        build_type="hybrid"
        ;;
    *)
        echo "Invalid choice, defaulting to CPU build."
        build_type="cpu"
        ;;
esac

# Keep option values intact when installation paths contain spaces.
cmake_options=()
protobuf_dir=${PROTOBUF_DIR:-}
if [[ -z "$protobuf_dir" ]]; then
    read -e -r -p "Enter Protobuf install prefix (leave empty to use system default): " protobuf_dir || true
fi
if [[ -n "$protobuf_dir" ]]; then
    cmake_options+=("-DPROTOBUF_DIR=$protobuf_dir")
fi

build_dir="build_$build_type"
if [[ "$build_type" == cpu ]]; then
    cmake_options+=(-DUSE_GPU_PHASE1=OFF -DUSE_GPU_PHASE2=OFF)
else
    cmake_options+=(-DUSE_GPU_PHASE1=ON -DUSE_GPU_PHASE2=ON)
    if [[ -n "${CUDAARCHS:-}" ]]; then
        cmake_options+=("-DCMAKE_CUDA_ARCHITECTURES=$CUDAARCHS")
    fi
fi

echo "Building $build_type version..."
cmake -S . -B "$build_dir" "${cmake_options[@]}"
cmake --build "$build_dir" --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}"

echo "------------------------------------------------"
echo "Build finished: $build_dir/HDGA"
