# BioToken

BioToken is a Python package for the tokenization and modeling of biological sequences with annotation support.

## Installation

```bash
# Create a virtual environment
conda create -n biofm-eval-env python=3.11
conda activate biofm-eval-env

# Clone biofm-eval repository
git clone https://github.com/m42-health/biofm-eval.git
cd biofm-eval

# Install biofm-eval package
pip install -e .

```

## Features

- `Annotator`: Functionality for annotating biological sequences with various features (e.g., variant information, genomic annotations, functional elements)
- `AnnotatedTokenizer`: Extended tokenizers that preserve biological annotations during tokenization
- `AnnotatedModel`: Models that can leverage the annotated tokens for enhanced performance

## Quick Start

### Embeddings for VCF Dataset

```python
from biofm_eval import AnnotatedModel, AnnotationTokenizer, Embedder, VCFConverter

import torch

MODEL_PATH = "/models_gfm/variant_paper/mistral/no-tweaked-mistral-mistral-265m-vlw-annohg1000-6k-step063489-chngont2"
TOKENIZER_PATH = "/models_gfm/variant_paper/mistral/no-tweaked-mistral-mistral-265m-vlw-annohg1000-6k-step063489-chngont2"

# Load model and tokenizer
model = AnnotatedModel.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
)
tokenizer = AnnotationTokenizer.from_pretrained(
    TOKENIZER_PATH,
)

embedder = Embedder(model, tokenizer)

# Load data
vcf_converter = VCFConverter(
    gene_annotation_path="/data/pretrain/genomics/gencode.v38.annotation.gff3",
    reference_genome_path="/data/pretrain/genomics/hg38_reference/GCA_000001405.15_GRCh38_no_alt_plus_hs38d1_analysis_set.fna"
)

# Create a huggingface dataset of variants using BioTokens
annotated_dataset = vcf_converter.vcf_to_annotated_dataset(
    vcf_path = '/data/pretrain/genomics/genome1000_corrected/HG01779_b.vcf.gz', 
    max_variants=200 # Set to None to create annotated dataset comprising all the variants from vcf
)

# Extract BioFM embeddings for all variants
embeddings = embedder.get_dataset_embeddings(annotated_dataset)
print(embeddings)



# import vcf
# vcf_reader = vcf.Reader(filename='/data/pretrain/genomics/genome1000_corrected/HG01779_b.vcf.gz')

# for variant in vcf_reader:
#     record = vcf_converter.annotate_snp_record(variant)
#     print(record)
#     if True:
#         break


```



## Documentation

TODO

## License

This project is licensed under the (TODO) License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 