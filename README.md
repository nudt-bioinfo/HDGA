# HDGA

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

HDGA (**H**eterogeneous **D**ifference-based **G**raph **A**ligner) is a high-performance sequence-to-graph alignment tool designed for heterogeneous CPU–GPU systems.  HDGA introduces a CPU–GPU collaborative parallel framework with a heuristic scheduling strategy to enable load-balanced execution across heterogeneous hardware. On top of this system design, it incorporates a difference-based recurrence formulation that maintains high SIMD parallelism regardless of read length, as well as an adaptive banded dynamic programming strategy that reduces unnecessary computation while preserving optimal alignment accuracy. HDGA builds upon the [PaSGAL](https://github.com/ParBLiSS/PaSGAL) framework for input preprocessing and output formatting. 

<p align="center">
  <img src="workflow.svg" width="100%">
  <br>
  <em>Fig: Workflow of HDGA.</em>
</p>



## Requirements

HDGA requires the following environment and libraries to build and run properly.

### Hardware Requirements

- **CPU**: AVX2 support required (tested on Intel Xeon Gold 6230)
- **GPU**: NVIDIA GPU with CUDA support (tested on RTX A6000)
- **Memory**: 64 GB RAM or higher recommended (256 GB used in our experiments)

HDGA is optimized for the `AVX2` instruction set. Please ensure your CPU supports AVX2 (check via `/proc/cpuinfo` on Linux).

---

### Software Requirements

- **Operating System**: Ubuntu 20.04 LTS
- **CUDA Toolkit**: 12.2
- **NVIDIA Driver**: 535.183.01
- **Compiler**:
  - Intel oneAPI DPC++/C++ Compiler (recommended)
  - or GCC with C++17 support (AVX2 enabled)
- **CMake**: ≥ 3.1

---

### Required Libraries and Tools

- [Google Protobuf](https://github.com/protocolbuffers/protobuf) (can also be installed via conda)
- vg toolkit (v1.70)
- ART (MountRainier, 2016-06-05)
- PBSIM (v1.0.3)

---

### Notes

- Intel oneAPI compiler is recommended for better SIMD optimization on Intel CPUs.
- GPU acceleration requires a CUDA-capable device with sufficient memory (≥16 GB recommended).

---

## Installation and Deployment
### Build
```sh
git clone https://github.com/nudt-bioinfo/HDGA.git
cd HDGA
./build.sh
```

HDGA provides an interactive build script to compile different execution modes, you will be prompted to select the build target:

- **CPU version** (AVX2 required)
- **GPU version** (CUDA required)
- **Hybrid version** (CPU–GPU collaborative, recommended)

The script will automatically configure and compile the selected components.

### Run

HDGA provides a unified execution script:
```sh
./run.sh
```
The script supports interactive configuration of the following options:

- Execution mode:
  - CPU
  - GPU
  - CPU–GPU collaborative (recommended)

- Number of CPU threads
- Input datasets:
  - Reference graph (e.g., BRCA1.vg, LRC.vg)
  - Read set
- Output path
- Alignment configuration:
  - Enable/disable adaptive banded mode

## Quickstart
The following example demonstrates how to run HDGA using the provided script and built-in datasets:
```sh
./run.sh
========================================
        HDGA Interactive Runner         
========================================

Select execution mode:
1) CPU
2) GPU
3) CPU-GPU Collaborative (recommended)
Enter choice [1-3]: 3

Enter number of CPU threads (e.g., 16): 16

Enter path to reference graph (.vg): ../data/graph/BRCA1.vg

Enter path to reads file (.fastq): ../data/reads/reads.fa

Enter output file prefix: outputfile

Enable adaptive banded mode?
1) Yes (recommended for reads ≥ 1kbp)
2) No
Enter choice [1-2]: 2

========================================
Running HDGA with the following config:
Mode: hybrid
Threads: 16
Graph: ../data/graph/BRCA1.vg
Reads: ../data/reads/reads.fa
Output: outputfile
Banded: 
========================================

Done.

```

<!-- ## Datasets
### Reference genome
- Human genome GRCh38.p14: https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000001405.40/

- E.coli genome (Strain:ATCC11775, ASM369716v2): https://gtdb.ecogenomic.org/genome?gid=GCF_003697165.2

### VCF files
VCF files for GRCh38 in 1000 Genomes Project: https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000_genomes_project/release/20190312_biallelic_SNV_and_INDEL/

### Reference graph
The reference graph is constructed using [vg](https://anaconda.org/bioconda/vg) from the reference genome and VCF files.

| graph         | reference genome (chr – region)         | vcf                                                          |
| ------------- | --------------------------------------- | ------------------------------------------------------------ |
| LRC           | GRCh38.p14 Chr19: 54025633-55084318     | the same region in ALL.chr19.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased.vcf.gz |
| MHC           | GRCh38.p14 Chr6: 28510119-33480577      | the same region in ALL.chr6.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased.vcf.gz |
| linear E.coli | GCF_003697165.2_ASM369716v2_genomic.fna | \                                                            |

### Readset
The Long Pacbio readset were simulated using [pbsim](https://anaconda.org/bioconda/pbsim). -->