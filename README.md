# IDEA

**IDEA: an interpretable framework for spatial transcriptomics niche identification and cell-type composition inference**

IDEA is a unified and scalable framework for multilevel analysis of spatial transcriptomics data. It integrates spatial niche identification, cell-type composition inference, and gene-level interpretation within a common modeling framework.

IDEA supports analyses across heterogeneous slices, platforms, spatial resolutions, and large-scale spatial transcriptomics datasets.

## Overview

Spatial niche identification, cell-type composition inference, and gene-level interpretation provide complementary views of tissue organization, but these tasks are often addressed separately.

IDEA provides three main analysis modules:

* Spatial niche identification and cross-slice integration
* Cell-type composition inference using single-cell RNA-seq references
* Post hoc interpretation for identifying niche- and cell-type-associated genes


## Installation

Clone the repository:

```bash
git clone https://github.com/REN-J-L/IDEA.git
cd IDEA
```

Create a conda environment:

```bash
conda create -n idea python=3.10
conda activate idea
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Installation time

Installation typically takes approximately 5–10 minutes, depending on internet speed and the local software environment.

## System requirements

IDEA has been tested on:

- Microsoft Windows 11, version 25H2 (OS build 26200.9445)
- Ubuntu 24.04.4 LTS (Noble Numbat)
- Python 3.10
- NVIDIA CUDA-compatible GPUs

## Using IDEA on your own data

IDEA accepts spatial transcriptomics data stored as an AnnData object.

For spatial niche identification, the input should contain:

* a gene-expression matrix in `adata.X`;
* spatial coordinates in `adata.obsm`;
* sample identifiers for multi-slice analyses, where applicable.

For cell-type composition inference, IDEA additionally requires a matched scRNA-seq reference with cell-type annotations.

Spatial and reference datasets should contain a common set of genes.

Users are encouraged to follow the corresponding tutorial for each analysis task.

## Tutorials

Example workflows are provided for:

* Single-slice spatial niche identification and niche-associated gene analysis
* Multi-slice spatial niche identification and niche-associated gene analysis
* Cell-type composition inference and cell-type-associated gene analysis for high-resolution datasets
* Cell-type composition inference and cell-type-associated gene analysis for low-resolution datasets

See the [`tutorial`](./tutorial) directory for detailed examples.

## Datasets

The datasets used in the IDEA study were originally generated and published by previous studies. Processed datasets used to reproduce the analyses are available separately.

Dataset descriptions and download links are provided in the corresponding data repository:

**Data repository:** [Zenodo](https://doi.org/10.5281/zenodo.22145448)

No newly generated experimental sequencing data are included in the repository.

## Expected runtime

Approximate runtimes for the tutorial workflows were measured on an NVIDIA GeForce RTX 4090 GPU. Runtime may vary depending on hardware configuration, dataset size, and preprocessing settings.

| Tutorial | Dataset size | Preprocessing | Model training | Clustering / resolution search | Gene-level interpretation |
| --- | --- | ---: | ---: | ---: | ---: |
| Low-resolution cell-type composition inference | 71 spatial units | ~10 min | ~2 min | – | ~2 s |
| High-resolution cell-type composition inference | 115,165 spatial units | ~3 min | ~5 min | – | ~13 min |
| Single-slice spatial niche identification | 123,836 spatial units, 1,022 genes | –  | ~6 min | ~9 min | ~5 min |
| Multi-slice spatial niche identification | 714,252 spatial units, 299 genes | – | ~12 min | ~60 min | ~6 min |


## License

This project is released under the [LICENSE](./LICENSE) included in this repository.

## Contact

For questions, bug reports, or feature requests, please open an issue in this repository.
