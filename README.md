# BioFM-Eval: Biologically Informed Tokenization and Embedding Extraction

BioFM-Eval is a Python package for inference and embedding extraction from genomic sequences. It features biologically informed tokenization (BioToken) and annotation-based sequence processing for downstream analysis.

![BioFM](biotoken_biofm.png)

## Contents

- [System Requirements](#system-requirements)
    - [Hardware Requirements](#hardware-requirements)
    - [Software Requirements](#software-requirements)
- [Installation](#installation)
- [Features](#features)
- [Quick Start](#quick-start)
    - [BioFM Model on Hugging Face 🤗](#biofm-model-on-hugging-face)
    - [Creating Variant Embeddings with BioFM](#creating-variant-embeddings-with-biofm)
    - [Sequence Embeddings with BioFM](#sequence-embeddings-with-biofm)
    - [Generation with BioFM](#generation-with-biofm)
- [License](#license)
- [Contribution](#contribution)
- [Citation](#citation)

## System Requirements
Before installing `BioFM-Eval`, please ensure your system meets the following requirements.

### Hardware Requirements
The `BioFM-Eval` package is designed to run on a standard computer with sufficient RAM for BioFM model inference, even without a GPU. The package has been successfully tested on the following hardware configurations:

- MacBook with M2 Pro chip and 16GB unified memory
- Linux system with Intel Xeon processor, 128GB RAM, and 1x H100 GPU


### Software Requirements
`BioFM-Eval` has been tested on the following operating systems:

- Ubuntu 22.04
- macOS Sequoia

## Installation

```bash
# Create a virtual environment
conda create -n biofm-eval-env python=3.11
conda activate biofm-eval-env

# Clone biofm-eval repository
git clone https://github.com/m42-health/biofm-eval.git
cd biofm-eval

# Install biofm-eval package along with all its dependencies
# Installation should take under 60 seconds on a Macbook
pip install -e .

```

### Using UV (an extremely fast Python package installer and resolver)

```bash
# Install uv
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# On Windows
# irm https://astral.sh/uv/install.ps1 | iex

git clone https://github.com/m42-health/biofm-eval.git
cd biofm-eval

# Install biofm-eval package along with all its dependencies using uv
uv sync
```

## Features

- `Annotator`: Enables annotation of biological sequences with features such as variant information, genomic annotations, and functional elements.
- `AnnotatedTokenizer`: A biologically informed tokenizer (BioToken) that preserves annotations during tokenization for improved sequence representation.
- `AnnotatedModel`: Supports extracting embeddings from annotated tokens using models like BioFM, allowing downstream applications to effectively utilize biological context.

## Quick Start

### BioFM Model on Hugging Face 🤗
The BioFM model is available on [Hugging Face](https://huggingface.co/m42-health/BioFM-265M).

This version has 265 million parameters and can run efficiently without requiring a GPU.

### Creating Variant Embeddings with BioFM

This guide will help you quickly generate BioFM embeddings for the variants in your VCF file. These embeddings are created using the method described in our publication. The following steps provide a high-level overview of the embedding extraction process.

- For decoder-only models like BioFM, embeddings are extracted using upstream (before the variant) and downstream (after the variant) sequences to ensure consistency.
- A mutated upstream sequence and a mutated downstream sequence are constructed, both ending with the variant and having a length of half the evaluation context size.
- The downstream sequence is reverse complemented before extracting embeddings to align with the reference strand.
- The upstream and downstream reference sequences are averaged, and the upstream and downstream mutated sequences are averaged.
- The two averaged vectors (reference and mutated) are concatenated to form the final embedding.
- This approach ensures equal context availability for all models and accounts for the causal nature of decoder-only architectures.

```python
from biofm_eval import AnnotatedModel, AnnotationTokenizer, Embedder, VCFConverter
import torch

# Define paths to the pre-trained BioFM model and tokenizer
MODEL_PATH = "m42-health/BioFM-265M"
TOKENIZER_PATH = "m42-health/BioFM-265M"

# Load the pre-trained BioFM model and BioToken tokenizer
model = AnnotatedModel.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
)
tokenizer = AnnotationTokenizer.from_pretrained(TOKENIZER_PATH)

# Initialize the embedder using the model and tokenizer
embedder = Embedder(model, tokenizer)

# Set up the VCF converter with paths to gene annotations and reference genome
vcf_converter = VCFConverter(
    gene_annotation_path="PATH/TO/gencode.v38.annotation.gff3",
    reference_genome_path="PATH/TO/hg38_reference/GCA_000001405.15_GRCh38_no_alt_plus_hs38d1_analysis_set.fna"
)

# Convert a VCF file into an annotated dataset using BioTokens
annotated_dataset = vcf_converter.vcf_to_annotated_dataset(
    vcf_path = 'PATH/TO/genome1000_corrected/HG01779_b.vcf.gz', 
    max_variants=200 # Set to None to process all variants in the VCF file
)

# Extract BioFM embeddings for all annotated variants
embeddings = embedder.get_dataset_embeddings(annotated_dataset)
print(embeddings)

# Example output (dict):
# {
#     'embeddings': array of shape (num_variants, 2*embedding_dim),  # Numeric embeddings for each variant
#     'labels': array of shape (num_variants,)  # Present only during supervised embedding extraction
# }
# Note that num_variants may be less than max_variants because of filtering and validity checks.

```
The embedding extraction code snippet above should take less than 30 seconds to process 200 variants.

- Sample reference genome fasta file: [download link](https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000001405.26/)
- Gene annotation file: [download_link](https://www.gencodegenes.org/human/release_38.html)
- Sample vcf file from 1000 Genomes data: [download_link](https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000_genomes_project/release/20190312_biallelic_SNV_and_INDEL/)

### Sequence Embeddings with BioFM
Embeddings for input DNA sequences can be generated for downstream tasks.

```python
from biofm_eval import AnnotatedModel, AnnotationTokenizer, Embedder
import torch

# Define paths to the pre-trained BioFM model and tokenizer
MODEL_PATH = "m42-health/BioFM-265M"
TOKENIZER_PATH = "m42-health/BioFM-265M"

# Load the pre-trained BioFM model and BioToken tokenizer
model = AnnotatedModel.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
)
tokenizer = AnnotationTokenizer.from_pretrained(TOKENIZER_PATH)

# Initialize the embedder using the model and tokenizer
embedder = Embedder(model, tokenizer)


# Generate sequence embedding
input_sequences = ['AGCT', 'GACTGCA']
sequence_embedding = embedder.get_sequence_embeddings(input_sequences)
print(f'Embedding dimension: {sequence_embedding.shape}')

# Embedding are extracted from the last token for each sequence
# Example output: torch.tensor of shape (num_sequences, embedding_dim) 

```

### Prediction of Variant Effect from VCF File

```python
from biofm_eval import AnnotatedModel, AnnotationTokenizer, Embedder, VCFConverter
import torch

# Define paths to the pre-trained BioFM model and tokenizer
MODEL_PATH = "m42-health/BioFM-265M"
TOKENIZER_PATH = "m42-health/BioFM-265M"
DATASET_PATH = "m42-health/variant-benchmark"

# Select one of expression, coding_pathogenicity, non_coding_pathogenicity, common_vs_rare, meqtl, sqtl
DATASET_SUBSET = "expression"
# Load the pre-trained BioFM model and BioToken tokenizer
model = AnnotatedModel.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
)
tokenizer = AnnotationTokenizer.from_pretrained(TOKENIZER_PATH)
dataset_dict = load_dataset(DATASET_PATH, DATASET_SUBSET)

# Initialize the embedder using the model and tokenizer
embedder = Embedder(model, tokenizer)

# Set up the VCF converter with paths to gene annotations and reference genome
vcf_converter = VCFConverter(
    gene_annotation_path="PATH/TO/gencode.v38.annotation.gff3",
    reference_genome_path="PATH/TO/hg38_reference/GCA_000001405.15_GRCh38_no_alt_plus_hs38d1_analysis_set.fna"
)

# Convert a VCF file into an annotated dataset using BioTokens
annotated_dataset = vcf_converter.vcf_to_annotated_dataset(
    vcf_path = 'PATH/TO/genome1000_corrected/HG01779_b.vcf.gz', 
    max_variants=200 # Set to None to process all variants in the VCF file
)

train_dataset = dataset_dict['train']
dd = {
    'train': dataset_dict['train'],
    'test': annotated_dataset
}
result = embedder.linear_probing(
        dd,
        batch_size=32
)
print(result.y_true.shape, result.y_pred.shape, result.y_pred_proba.shape)
```


### Generation with BioFM
BioFM can generate genomic sequences based on input DNA prompts.


```python
from biofm_eval import AnnotatedModel, AnnotationTokenizer, Generator
import torch

# Define paths to the pre-trained BioFM model and tokenizer
MODEL_PATH = "m42-health/BioFM-265M"
TOKENIZER_PATH = "m42-health/BioFM-265M"

# Load the pre-trained BioFM model and BioToken tokenizer
model = AnnotatedModel.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
)
tokenizer = AnnotationTokenizer.from_pretrained(TOKENIZER_PATH)

# Initializing the generator using model and tokenizer
seq_generator = Generator(model, tokenizer)

# Generate DNA sequences
input_sequences = ['AGCT', 'GACTGCA']
output = seq_generator.generate(
    input_sequences, 
    max_new_tokens=10, 
    temperature=1.0, 
    do_sample=True, 
    top_k=4)

print(output)

# Example output: List[str] = ['AGCTACTCCCCTCC', 'GACTGCACCACTGTACT']

```

### Reproduction of the Variant Benchmark from the paper

```python
from biofm_eval import Annotator, AnnotatedModel, AnnotationTokenizer, Embedder, VCFConverter
from datasets import DatasetDict
import torch
from sklearn.metrics import roc_auc_score
import numpy as np

# Define paths to the pre-trained BioFM model and tokenizer
MODEL_PATH = "m42-health/BioFM-265M"
TOKENIZER_PATH = "m42-health/BioFM-265M"
DATASET_PATH = "m42-health/variant-benchmark"
DATASET_SUBSET = "expression"
# Load the pre-trained BioFM model and BioToken tokenizer
model = AnnotatedModel.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
)
tokenizer = AnnotationTokenizer.from_pretrained(TOKENIZER_PATH)
dataset_dict = load_dataset(DATASET_PATH, DATASET_SUBSET)
embedder = Embedder(model, tokenizer)

for i in range(11):
    dd = split_dataset_by_chrom(dataset_dict['train'], fold=i)

    result = embedder.linear_probing(
        dd,
        batch_size=32
    )
    print(f'fold={i}, AUC={roc_auc_score(result.y_true, result.y_pred_proba):.4f}')
```


## License

This project is licensed under CC BY-NC-4.0 - see the [LICENSE.md](LICENSE.md) file for details.

## Contribution

Contributions are welcome! Please feel free to submit a Pull Request. 

## Citation
If you find this repository useful, please consider giving a star and citation:
```
@article {Medvedev2025.03.27.645711,
    author = {Medvedev, Aleksandr and Viswanathan, Karthik and Kanithi, Praveenkumar and Vishniakov, Kirill and Munjal, Prateek and Christophe, Clement and Pimentel, Marco AF and Rajan, Ronnie and Khan, Shadab},
    title = {BioToken and BioFM - Biologically-Informed Tokenization Enables Accurate and Efficient Genomic Foundation Models},
    elocation-id = {2025.03.27.645711},
    year = {2025},
    doi = {10.1101/2025.03.27.645711},
    publisher = {Cold Spring Harbor Laboratory},
    URL = {https://www.biorxiv.org/content/early/2025/04/01/2025.03.27.645711},
    eprint = {https://www.biorxiv.org/content/early/2025/04/01/2025.03.27.645711.full.pdf},
    journal = {bioRxiv}
}
```

