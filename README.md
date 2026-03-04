# ESMC demo

## Install

```bash
conda env create -f environment.yml -n esmcenv
conda activate esmcenv
```

## Usage

```bash
python generate_embedding.py -i proteins.fasta -o embeddings.parquet -m esmc_300m
```