"""BioToken: A package for annotated tokenization and modeling of biological sequences."""

__version__ = "0.1.0"

# Import main classes for easy access
from biofm_eval.annotators import Annotator
from biofm_eval.tokenizers import AnnotationTokenizer
from biofm_eval.models import AnnotatedModel
from biofm_eval.embedder import Embedder
from biofm_eval.generator import Generator
from biofm_eval.data_utils import VCFConverter, split_dataset_by_chrom
# For direct imports
__all__ = [
    "Annotator",
    "AnnotationTokenizer",
    "AnnotatedModel",
    "Embedder",
    "Generator",
    "VCFConverter",
    "split_dataset_by_chrom",
]
