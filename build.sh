#!/bin/bash

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

# Protobuf path handling
protobuf_option=""
if [ -z "${PROTOBUF_DIR}" ]; then
    read -e -p "Enter Protobuf install prefix (leave empty to use system default): " protobuf_dir_input
    if [ -n "${protobuf_dir_input}" ]; then
        protobuf_option="-DPROTOBUF_DIR=${protobuf_dir_input}"
    fi
else
    protobuf_option="-DPROTOBUF_DIR=${PROTOBUF_DIR}"
fi

# Build options
if [ "$build_type" = "gpu" ]; then
    echo "Building GPU version..."
    build_dir="build_gpu"
    gpu_option="-DUSE_GPU_PHASE1=ON -DUSE_GPU_PHASE2=ON"
elif [ "$build_type" = "hybrid" ]; then
    echo "Building hybrid (CPU+GPU) version..."
    build_dir="build_hybrid"
    gpu_option="-DUSE_GPU_PHASE1=ON -DUSE_GPU_PHASE2=ON"
else
    echo "Building CPU version..."
    build_dir="build_cpu"
    gpu_option=""
fi

# Create build directory
if [ ! -d "$build_dir" ]; then
    mkdir "$build_dir"
fi

# Run cmake and make
cd "$build_dir" || exit

cmake .. $gpu_option $protobuf_option
make -j$(nproc)

# Return to parent directory
cd ..

echo "------------------------------------------------"
echo "Build finished: $build_dir"
