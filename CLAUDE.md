# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CropGBM (Crop Genomic Breeding machine) is a Python3 program for genomic data analysis that integrates data preprocessing, population structure analysis, SNP selection, phenotype prediction, and visualization using LightGBM.

## Commands

### Running the CLI
```bash
cropgbm [options]           # After installation via conda/pip
python cropgbm [options]   # During development (from repo root, ./cropgbm is the CLI script)
```

### Using Config Files
Many parameters can be set via a `.params` config file (see `testdata/configfile.params` for an example):
```bash
cropgbm -c ./testdata/configfile.params -o ./
```

### Running Tests
```bash
python run_test.py          # Runs all 20 integration tests
```

### Key CLI Workflows

**Preprocess genotype data:**
```bash
cropgbm -pg all --fileprefix ./testdata/genofile --fileformat ped
cropgbm -pg filter --fileprefix <path> --remove-sampleid-path <path>
```

**Preprocess phenotype data:**
```bash
cropgbm -pp --phefile-path <path> --phe-name <col> --phe-norm
cropgbm -pp --phefile-path <path> --phe-name <col> --phe-recode word2num
```

**Population structure analysis:**
```bash
cropgbm -s --genofile-path <path> --structure-plot --redim-mode pca --cluster-mode kmeans --n-clusters 30
```

**Model training/prediction:**
```bash
cropgbm -e -t --traingeno <path> --trainphe <path> --validgeno <path> --validphe <path>
cropgbm -e -cv --traingeno <path> --trainphe <path> --cv-nfold 5
cropgbm -e -p --testgeno <path> --modelfile-path <path>
```

**Feature selection (runs after training):**
```bash
cropgbm -e -t -sf --traingeno <path> --trainphe <path> --bygain-boxplot --min-gain 0.05
# -sf extracts SNP importance from the model trained by -t
```

## Architecture

The codebase has a **module-based architecture** with a central CLI dispatcher:

### Core Modules

| File | Responsibility |
|------|---------------|
| `cropgbm` | CLI entry point script using argparse; dispatches to modules based on flags (`-pg`, `-pp`, `-s`, `-e`) |
| `Parameters.py` | Configuration parsing (`import_config_params`), default value filling (`fill_params_by_default`), param validation (`check_params`) |
| `Engine.py` | LightGBM training (`lgb_train`), cross-validation (`lgb_cv`), prediction (`lgb_predict`), and iterative feature analysis (`lgb_iter_feature`) |
| `Feature.py` | Extracts SNP gain values from trained `.lgb_model` files (`extree_info`), summarizes by regression/multiclass (`exfeature_by_regression`, `exfeature_by_classification`) |
| `Structure.py` | Dimensionality reduction via PCA (`redim_pca`) or t-SNE (`redim_tsne`), clustering via KMeans or OPTICS |
| `Preprocessed_Geno.py` | Calls external `plink` tool for genotype QC, filtering, and recoding to 0/1/2 format |
| `Preprocessed_Pheno.py` | Phenotype normalization, recoding (word↔num), group extraction |
| `Visualize.py` | Plotting: heatmaps (`plot_heatmap`), structure scatter plots, histograms, heterozygosity/missing rate distributions |

### Data Flow

1. **Preprocessing** (`-pg`, `-pp`): Raw PLINK files → filtered genotype CSV; raw phenotype → normalized/recoded phenotype
2. **Structure** (`-s`): Genotype CSV → PCA/t-SNE reduced dimensions → KMeans/OPTICS clustering → visualization
3. **Engine** (`-e`): Genotype + phenotype CSV → LightGBM model (`.lgb_model` file) → predictions or feature importance

### Important Notes

- **plink dependency**: `Preprocessed_Geno.py` calls the external `plink` binary (must be in PATH or specified via `--plink-path`)
- **File formats**: Genotype uses CSV with sampleids as rows and SNPIDs as columns; phenotype uses CSV with sampleid index and `phe` column
- **LightGBM version constraint**: Requires `lightgbm >=3.3.0,<4.0.0` per `requirements` in README
- **Test data**: All test inputs are in `testdata/` directory; outputs are written to `preprocessed/`, `structure/`, and `engine/` folders
