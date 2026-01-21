from datasets import Dataset, DatasetDict
import vcf
import logging
from typing import List, Dict, Any, Optional, Union
from biofm_eval.annotators.base import Annotator
from pyfaidx import Fasta
from Bio.Seq import reverse_complement


class VCFConverter:
    def __init__(
        self,
        gene_annotation_path: str,
        reference_genome_path: str,
        context_size: int = 1024,
        add_chr_prefix: bool = False,
    ):
        """
        Convert a VCF file to an annotated dataset

        Parameters:
        -----------
        gene_annotation_path : str
            Path to the annotation file (in GFF/GTF format)
        reference_genome_path : str, optional
            Path to the reference genome in FASTA format.
            If not provided, sequences will be filled with 'N's.
        context_size : int, default=1024
            Size of the sequence context to extract around each variant.
        add_chr_prefix : bool, default=False
            If True, adds 'chr' prefix to chromosome names that don't have it.
        """

        self.anno_path = gene_annotation_path
        self.reference_genome_path = reference_genome_path
        self.reference_genome = Fasta(reference_genome_path)
        self.context_size = context_size
        self.add_chr_prefix = add_chr_prefix
        self.annotator = Annotator(
            annotation_path=self.anno_path, sequence_length=self.context_size
        )
        self.logger = logging.getLogger(__name__)

    def annotate_snp_record(self, variant: vcf.model._Record) -> Dict[str, Any]:
        """
        Annotate a SNP record
        """
        # Calculate positions for flanking sequence
        flank_size = self.context_size // 2
        start_pos = max(0, variant.POS - 1 - flank_size)

        # Get chromosome, optionally add chr prefix
        chrom = variant.CHROM
        if self.add_chr_prefix and not chrom.startswith("chr"):
            chrom = "chr" + chrom

        # Extract sequence from reference
        if chrom not in self.reference_genome:
            self.logger.warning(f"Chromosome {chrom} not found in reference genome")
            return None
        dna = str(
            self.reference_genome[chrom][start_pos : start_pos + self.context_size]
        )
        variant_idx = len(dna) // 2
        sequence_len = len(dna)
        assert dna[variant_idx] == variant.REF, f'dna[variant_idx]={dna[variant_idx-3:variant_idx+3]}, variant.REF={variant.REF}, variant.ALT={variant.ALT}'
        dna_alt_left = list(dna[1 : variant_idx + 1])
        dna_alt_left[-1] = str(variant.ALT[0])
        dna_alt_left = "".join(dna_alt_left)

        rc = reverse_complement(dna[variant_idx:])
        dna_alt_right = list(rc)
        dna_alt_right[-1] = reverse_complement(str(variant.ALT[0]))
        dna_alt_right = "".join(dna_alt_right)

        alt_left = self.annotator.annotate(
            {
                "chr": chrom,
                "start": start_pos + 1,
                "nt_seq": dna_alt_left,
                "mut_coords": [
                    {
                        "mut_start": len(dna_alt_left) - 1,
                        "mut_end": len(dna_alt_left),
                        "ref": variant.REF,
                        "alt": str(variant.ALT[0]),
                    }
                ],
                "is_reverse_complement": False,
            }
        )["nt_seq"]
        alt_right = self.annotator.annotate(
            {
                "chr": chrom,
                "start": start_pos + sequence_len // 2,
                "nt_seq": dna_alt_right,
                "mut_coords": [
                    {
                        "mut_start": len(dna_alt_right) - 1,
                        "mut_end": len(dna_alt_right),
                        "ref": reverse_complement(variant.REF),
                        "alt": reverse_complement(str(variant.ALT[0])),
                    }
                ],
                "is_reverse_complement": True,
            }
        )["nt_seq"]

        ref_left = self.annotator.annotate(
            {
                "chr": chrom,
                "start": start_pos + 1,
                "nt_seq": dna[1 : variant_idx + 1],
                "is_reverse_complement": False,
            }
        )["nt_seq"]

        ref_right = self.annotator.annotate(
            {
                "chr": chrom,
                "start": start_pos + sequence_len // 2,
                "nt_seq": rc,
                "is_reverse_complement": True,
            }
        )["nt_seq"]

        assert len(alt_left) == len(ref_left)
        assert len(alt_right) == len(
            ref_right
        ), f"alt_right)={alt_right}, ref_right)={ref_right}"
        return {
            "alt_left": alt_left,
            "alt_right": alt_right,
            "ref_left": ref_left,
            "ref_right": ref_right,
            "chromosome": chrom,
        }

    def vcf_to_annotated_dataset(
        self, vcf_path: str, max_variants: int = None
    ) -> Dataset:
        """
        Convert a VCF file to an annotated dataset

        Parameters:
        -----------
        vcf_path : str
            Path to the VCF file
        max_variants : int, default=None
            To limit the number of variants to process.
            Default value will process all the variants from vcf

        Returns:
        --------
        Dataset
            A Hugging Face dataset containing the annotated variants
        """

        # Open the VCF file with PyVCF
        try:
            vcf_reader = vcf.Reader(filename=vcf_path)
        except Exception as e:
            self.logger.error(f"Failed to open VCF file: {e}")
            raise

        records = []
        variant_count = 0
        snp_count = 0
        non_snp_count = 0

        for variant in vcf_reader:
            variant_count += 1

            # Check if variant is SNP (single nucleotide polymorphism)
            is_snp = len(variant.REF) == 1 and all(
                len(str(alt)) == 1 for alt in variant.ALT
            )

            # Skip non-SNPs if not explicitly included
            if not is_snp:
                non_snp_count += 1
                continue

            if is_snp:
                snp_count += 1

            try:
                annotated_record = self.annotate_snp_record(variant)
                if annotated_record is None:
                    self.logger.warning(f"Failed to annotate variant {variant}")
                    continue
                records.append(annotated_record)
            except Exception as e:
                self.logger.error(f"Failed to annotate variant {variant.ID}: {e}")
                continue

            if max_variants and (max_variants <= variant_count):
                break

        # Log statistics
        self.logger.info(
            f"Processed {variant_count} variants: {snp_count} SNPs, {non_snp_count} non-SNPs"
        )
        self.logger.info(f"Created {len(records)} annotated records")

        # Create a dataset from the records
        dataset = Dataset.from_list(records)

        return dataset


HUMAN_FOLD_SPLIT = {
    0: ['chr1', 'chr2'],
    1: ['chr3', 'chr4'],
    2: ['chr5', 'chr6'],
    3: ['chr7', 'chr8'],
    4: ['chr9', 'chr10'],
    5: ['chr11', 'chr12'],
    6: ['chr13', 'chr14'],
    7: ['chr15', 'chr16'],
    8: ['chr17', 'chr18'],
    9: ['chr19', 'chr20'],
    10: ['chr21', 'chr22', 'chrX'],
}

# Backwards compatibility alias
FOLD_SPLIT = HUMAN_FOLD_SPLIT

def get_fold_split(fold: Optional[int] = None, split_name: str = 'test',
    fold_split: Optional[Dict[int, List[str]]] = None,
) -> List[str]:
    if fold is None:
        fold = 0
    if fold_split is None:
        fold_split = HUMAN_FOLD_SPLIT
    if split_name == 'test':
        if fold not in fold_split:
            raise ValueError(f'Fold {fold} not found in fold_split. Available folds: {list(fold_split.keys())}')
        return fold_split[fold]
    else:
        raise ValueError(f'Unknown split name: {split_name}, should be test')


def split_dataset_by_chrom(dataset: Dataset, fold: Optional[int] = None,
    fold_split: Optional[Dict[int, List[str]]] = None,
    chrom_column: str = 'chrom') -> DatasetDict:
    """
    Split a dataset by chromosome

    Parameters:
    -----------
    dataset : Dataset
        The dataset to split.
    fold : int, optional
        Fold index for the test set (default: 0).
    fold_split : Dict[int, List[str]], optional
        Custom fold split configuration mapping fold indices to chromosome lists.
        If not provided, uses HUMAN_FOLD_SPLIT.
        Use generate_fold_split() to create a custom configuration for other organisms.
    chrom_column : str, default='chrom'
        Name of the column containing chromosome information.

    Returns:
    --------
    DatasetDict
        A DatasetDict with 'train' and 'test' splits.

    Example:
    --------
    >>> # For human data (default)
    >>> splits = split_dataset_by_chrom(dataset, fold=0)
    >>>
    >>> # For other organisms
    >>> chroms = dataset.unique('chromosome')
    >>> custom_split = generate_fold_split(chroms, n_folds=5)
    >>> splits = split_dataset_by_chrom(dataset, fold=0, fold_split=custom_split, chrom_column='chromosome')
    """
    test_chroms = get_fold_split(fold, 'test', fold_split)

    train_dataset = dataset.filter(
        lambda x: x[chrom_column] not in test_chroms,
        keep_in_memory=True
    )
    test_dataset = dataset.filter(
        lambda x: x[chrom_column] in test_chroms,
        keep_in_memory=True
)

    return DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })

def generate_fold_split(
    chromosomes: List[str],
    n_folds: int = 10,
    chromosomes_per_fold: Optional[int] = None,
) -> Dict[int, List[str]]:
    """
    Generate a fold split configuration from a list of chromosomes.

    Parameters:
    -----------
    chromosomes : List[str]
        List of chromosome names present in the dataset.
    n_folds : int, default=10
        Number of folds to create.
    chromosomes_per_fold : int, optional
        Number of chromosomes per fold.

    Returns:
    --------
    Dict[int, List[str]]
        A dictionary mapping fold indices to lists of chromosome names.

    Example:
    --------
    >>> chroms = ['I', 'II', 'III', 'IV', 'V', 'X']
    >>> fold_split = generate_fold_split(chroms, n_folds=3)
    >>> # Returns: {0: ['I', 'II'], 1: ['III', 'IV'], 2: ['V', 'X']}
    """
    unique_chroms = sorted(set(chromosomes))
    n_chroms = len(unique_chroms)

    if chromosomes_per_fold is not None:
        fold_split = {}
        for i in range(n_folds):
            start_idx = i * chromosomes_per_fold
            end_idx = min(start_idx + chromosomes_per_fold, n_chroms)
            if start_idx < n_chroms:
                fold_split[i] = unique_chroms[start_idx:end_idx]
        return fold_split

    base_size = n_chroms // n_folds
    remainder = n_chroms % n_folds

    fold_split = {}
    idx = 0
    for fold in range(n_folds):
        fold_size = base_size + (1 if fold < remainder else 0)
        if fold_size > 0:
            fold_split[fold] = unique_chroms[idx:idx + fold_size]
            idx += fold_size

    return fold_split
