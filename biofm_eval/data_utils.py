from datasets import Dataset
import vcf
import logging
from typing import List, Dict, Any, Optional, Union
from biofm_eval.annotators.base import Annotator
from pyfaidx import Fasta
from Bio.Seq import reverse_complement


class VCFConverter:
    def __init__(self, anno_path: str, reference_genome: str, context_size: int = 1024):
        self.anno_path = anno_path
        self.reference_genome_path = reference_genome
        self.reference_genome = Fasta(reference_genome)
        self.context_size = context_size
        self.annotator = Annotator(
            annotation_path=anno_path, sequence_length=context_size
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
        assert dna[variant_idx] == variant.REF
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
        anno_path : str
            Path to the annotation file (in GFF/GTF format)
        reference_genome : str, optional
            Path to the reference genome in FASTA format.
            If not provided, sequences will be filled with 'N's.
        context_size : int, default=1024
            Size of the sequence context to extract around each variant.

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
                    self.logger.warning(f"Failed to annotate variant {variant.ID}")
                    continue
                records.append(annotated_record)
            except Exception as e:
                self.logger.error(f"Failed to annotate variant {variant.ID}: {e}")
                continue

            if max_variants and (max_variants == variant_count):
                break

        # Log statistics
        self.logger.info(
            f"Processed {variant_count} variants: {snp_count} SNPs, {non_snp_count} non-SNPs"
        )
        self.logger.info(f"Created {len(records)} annotated records")

        # Create a dataset from the records
        dataset = Dataset.from_list(records)

        return dataset
