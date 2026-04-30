#!/bin/bash

set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT_DIR="$SCRIPT_DIR"

resolve_path() {
    local input_path="$1"
    if [ -z "$input_path" ]; then
        echo ""
    elif [[ "$input_path" = /* ]]; then
        echo "$input_path"
    else
        echo "$ROOT_DIR/$input_path"
    fi
}

echo "========================================"
echo "           HDGA Interactive Run         "
echo "========================================"

echo ""
echo "Select run mode:"
echo "1) CPU"
echo "2) GPU"
echo "3) Hybrid (CPU+GPU collaborative)"
read -p "Enter choice [1-3]: " mode_choice

case "$mode_choice" in
    1)
        mode="cpu"
        build_dir="build_cpu"
        exe="./HDGA"
        extra_args=""
        ;;
    2)
        mode="gpu"
        build_dir="build_gpu"
        exe="./HDGA"
        extra_args=""
        ;;
    3)
        mode="hybrid"
        build_dir="build_hybrid"
        exe="./HDGA"
        extra_args="--hybrid-cpu-gpu --cpu-speed 1 --gpu-speed 3"
        ;;
    *)
        echo "Invalid choice, defaulting to CPU."
        mode="cpu"
        build_dir="build_cpu"
        exe="./HDGA"
        extra_args=""
        ;;
esac

echo ""
read -p "Enter number of CPU threads (default 16): " threads
threads=${threads:-16}

echo ""
read -e -p "Enter path to reference graph (.vg): " graph

echo ""
read -e -p "Enter path to reads file (.fastq/.fa): " reads

echo ""
read -e -p "Enter output file prefix (default hdga_output): " output
output=${output:-hdga_output}

echo ""
echo "Enable adaptive banded mode?"
echo "1) Yes (recommended for long reads)"
echo "2) No"
read -p "Enter choice [1-2]: " band_choice

if [ "$band_choice" = "1" ]; then
    band_flag=""
else
    band_flag="--disable-banded"
fi

graph=$(resolve_path "$graph")
reads=$(resolve_path "$reads")

build_path="$ROOT_DIR/$build_dir"
exe_path="$build_path/HDGA"

if [ ! -d "$build_path" ]; then
    echo "Build directory not found: $build_path"
    echo "Please run ./build.sh first."
    exit 1
fi

if [ ! -x "$exe_path" ]; then
    echo "Executable not found: $exe_path"
    echo "Please build the selected mode first."
    exit 1
fi

if [ ! -f "$graph" ]; then
    echo "Reference graph not accessible: $graph"
    exit 1
fi

if [ ! -f "$reads" ]; then
    echo "Reads file not accessible: $reads"
    exit 1
fi

echo ""
echo "========================================"
echo "Running HDGA with the following config:"
echo "Mode: $mode"
echo "Build dir: $build_path"
echo "Threads: $threads"
echo "Graph: $graph"
echo "Reads: $reads"
echo "Output: $output"
echo "Banded: ${band_flag:---disable-banded}"
if [ "$mode" = "hybrid" ]; then
    echo "Hybrid weights: cpu-speed 1, gpu-speed 3"
fi
echo "========================================"
echo ""

cd "$build_path"
$exe_path -m vg \
    -r "$graph" \
    -q "$reads" \
    -t "$threads" \
    -o "$output" \
    $extra_args \
    $band_flag

echo ""
echo "Done."