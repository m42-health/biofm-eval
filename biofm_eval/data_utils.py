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
        """

        self.anno_path = gene_annotation_path
        self.reference_genome_path = reference_genome_path
        self.reference_genome = Fasta(reference_genome_path)
        self.context_size = context_size
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

        # Get chromosome, add chr prefix if not present
        chrom = variant.CHROM
        if not chrom.startswith("chr"):
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


FOLD_SPLIT = {
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


def get_fold_split(fold: Optional[int] = None, split_name: str = 'test') -> List[str]:
    if fold is None:
        fold = 0
    if split_name == 'test':
        return FOLD_SPLIT[fold]
    else:
        raise ValueError(f'Unknown split name: {split_name}, should be test')
    

def split_dataset_by_chrom(dataset: Dataset, fold: Optional[int] = None) -> DatasetDict:
    """
    Split a dataset by chromosome
    """

    train_dataset = dataset.filter(lambda x: x['chrom'] not in get_fold_split(fold, 'test'), keep_in_memory=True)
    test_dataset = dataset.filter(lambda x: x['chrom'] in get_fold_split(fold, 'test'), keep_in_memory=True)

    return DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })
