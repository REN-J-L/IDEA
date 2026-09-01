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

**Data repository:** [Zenodo: Datasets](https://zenodo.org/records/22145448)

No newly generated experimental sequencing data are included in the repository.


## License

This project is released under the [LICENSE](./LICENSE) included in this repository.

## Contact

For questions, bug reports, or feature requests, please open an issue in this repository.
