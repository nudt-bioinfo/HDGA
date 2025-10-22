# DGA

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

DGA (**D**ifference-based **G**raph **A**ligner) is a vectorized sequence-to-graph alignment tool based on difference recurrence that could produce optimal alignment results. We first introduce a novel difference recurrence instead of the classical Smith-Waterman algorithm. By doing so, our algorithm can keep the number of bits required to store each cell score within 8 even when aligning long reads, and subsequently maintain the highest parallelism of vectorization. In addition, we employ an adaptive banded approach that significantly reduces the number of cells that require computation.

<!-- ![workflow of DGA](workflow.svg) -->
<p align="center">
  <img src="workflow.svg" width="100%">
  <br>
  <em>Fig: Workflow of DGA.</em>
</p>


## Code Origin and License Notice
This project builds upon the open-source implementation of [PaSGAL](https://github.com/ParBLiSS/PaSGAL.git)
, from which we reuse components related to command-line argument parsing, input preprocessing, and output formatting.

The algorithmic core and main implementation in this repository were independently developed by us.

The reused parts from PaSGAL are licensed under the MIT License
.

Our modifications and new implementations are also released under the MIT License.

## Build & Usage
The overall dependencies, build process and usage of DGA follow those of [PaSGAL](https://github.com/ParBLiSS/PaSGAL.git).
Please refer to its documentation for detailed setup and compilation instructions.

Our implementation is optimized for the AVX2 instruction set. Therefore, please make sure your CPU supports AVX2 before compilation. The supported instruction set is listed in the `flags` section of the `/proc/cpuinfo` file on the Linux OS.

We recommend using the Intel oneAPI C++ compiler (icpx) for compilation, as it provides better optimization for Intel SIMD instruction sets.

### Quickstart
The following provides a quick example of how to build and run DGA.

```sh
# download and compile
git clone https://github.com/nudt-bioinfo/DGA.git
cd DGA
mkdir build && cd build
cmake -DSIMD_SUPPORT=avx2 -DCMAKE_CXX_COMPILER=icpx  ../DGA
make -j4

# run
./DGA -m vg -r ../data/graph/LRC.vg -q ../data/reads/LRC_0001.fastq -t 16 -o outputfile
```


## Datasets
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
The Long Pacbio readset were simulated using [pbsim](https://anaconda.org/bioconda/pbsim).