import pandas as pd
import numpy as np
from typing import Dict, Any
import re


ANNOTATION_MAP = {
    "CDS": ("[START_CDS]", "[END_CDS]"),
    "transcript": ("[START_TRANSCRIPT]", "[END_TRANSCRIPT]"),
    "exon": ("[START_EXON]", "[END_EXON]"),
}


VARIANT_MAP = {"A": "Â", "C": "Ĉ", "G": "Ĝ", "T": "Ṱ", "N": "N"}


def encode_variants(sample: Dict[str, Any], del_mut_coords=True, mut_encoding_type='same') -> Dict[str, Any]:
    # mut_coords = [{'mut_start': i, 'mut_end': i+len(alt[0]), 'alt': alt[0], 'ref': ref}]
    if "mut_coords" not in sample:
        return sample
    seq = list(sample["nt_seq"])
    for coords in sample["mut_coords"]:
        alt = sample["nt_seq"][coords["mut_start"] : coords["mut_end"]]
        if alt != coords['alt']:
            print(f'ALT: {alt}, coords[alt]: {coords["alt"]}, coords: {coords}')
            print(sample['mut_coords'])
            print(sample['nt_seq'])
            assert alt == coords['alt']
        try:
            if mut_encoding_type == 'same': 
                seq[coords["mut_start"] : coords["mut_end"]] = [VARIANT_MAP[s] for s in alt]
            else:
                seq[coords["mut_start"] : coords["mut_end"]] = [s for s in alt]
        except KeyError:
            print(f'alt={alt} all coords are {sample["mut_coords"]}')
        # print(f'We replaced {len(alt)} nucleotides, {len(coords["alt"])} with {[VARIANT_MAP[s] for s in alt]}')
    if del_mut_coords:
        del sample["mut_coords"]
    sample["nt_seq"] = "".join(seq)
    return sample


class Annotator:
    def __init__(
        self,
        annotation_path: str,
        mut_encoding_type: str = 'same',
        sequence_length: int = None,
        canonical_only: bool = False
    ):
        self.mut_encoding_type = mut_encoding_type
        self.annotation_path = annotation_path
        self.sequence_length = sequence_length
        self.canonical_only = canonical_only
        self._read_annotations()

    def _read_annotations(self):
        annotations = pd.read_csv(
            self.annotation_path,
            sep="\s+",
            comment="#",
            header=None,
            names=[
                "chrom",
                "source",
                "annotation_type",
                "start",
                "end",
                "score",
                "strand",
                "phase",
                "attributes",
            ],
        )
        at_mask = annotations["annotation_type"].isin(["CDS", "exon", "transcript"])
        if self.canonical_only:
            canonical_mask = np.array(
                ["Ensembl_canonical" in s for s in annotations["attributes"]]
            )
            annotations = annotations.loc[at_mask & canonical_mask, :]
        else:
            annotations = annotations.loc[at_mask, :]
        print(
            f"After filtering, we have total of {len(annotations)} canonical annotations of CDS, exon, transcript"
        )
        # example is ID=ENSG00000223972.5;gene_id=ENSG00000223972.5;gene_type=transcribed_unprocessed_pseudogene;gene_name=DDX11L1;level=2;hgnc_id=HGNC:37102;havana_gene=OTTHUMG00000000961.2
        annotations['gene_name'] = [re.search(r'gene_name=([^;]+)', x).group(1) for x in annotations['attributes']]
        annotations = annotations.groupby(by=["chrom", "strand"])
        self.annotations = {}
        for key, group in annotations:
            # TODO: REMOVE LATER
            group = group.drop_duplicates(subset=['annotation_type', 'start', 'end'])
            self.annotations[key] = group

    def annotate(self, item):
        item = encode_variants(item, del_mut_coords=False, mut_encoding_type=self.mut_encoding_type)
        item = self._annotate(item)
        return {"nt_seq": item["nt_seq"]}

    def _annotate(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        # print(sample)
        annotations = self.annotations[
            (sample["chr"], "-" if sample.get("is_reverse_complement", True) else "+")
        ]
        if self.sequence_length is None:
            self.sequence_length = len(sample["nt_seq"])
            
        left, right = sample["start"], sample["start"] + self.sequence_length

        relevant = annotations[
            ((left + 1 <= annotations["start"]) & (annotations["start"] <= right + 1))
            | ((left + 1 <= annotations["end"]) & (annotations["end"] < right + 1))
        ]

        if len(relevant) == 0:
            sample["nt_seq"] = " ".join(list(sample["nt_seq"]))
            return sample
        
        relevant = relevant.groupby("annotation_type")
        tokens_to_insert = []
        for at, group in relevant:
            if at not in ANNOTATION_MAP:
                continue
            for start, end, strand in zip(
                group["start"], group["end"], group["strand"]
            ):
                if strand == "+":
                    rel_start, rel_end = (
                        start - sample["start"] - 1,
                        end - sample["start"],
                    )
                else:
                    rel_start, rel_end = (
                        self.sequence_length - (end - sample["start"]),
                        self.sequence_length - (start - sample["start"] - 1),
                    )
                start_shift = sum(
                    [
                        len(c["ref"]) - len(c["alt"])
                        for c in sample.get("mut_coords", [])
                        if c["mut_start"] < rel_start
                    ]
                )
                end_shift = sum(
                    [
                        len(c["ref"]) - len(c["alt"])
                        for c in sample.get("mut_coords", [])
                        if c["mut_start"] < rel_end
                    ]
                )
                if rel_start + start_shift >= 0:
                    token = ANNOTATION_MAP[at][0]
                    tokens_to_insert.append(
                        {"pos": rel_start + start_shift, "token": token}
                    )
                if rel_end + end_shift < len(sample["nt_seq"]):
                    token = ANNOTATION_MAP[at][1]
                    tokens_to_insert.append(
                        {"pos": rel_end + end_shift, "token": token}
                    )

        if len(tokens_to_insert) == 0:
            sample["nt_seq"] = " ".join(list(sample["nt_seq"]))
            return sample
        
        tokens_to_insert = sorted(tokens_to_insert, key=lambda x: x["pos"])
        last_pos = tokens_to_insert[0]["pos"]
        new_seq = [" ".join(list(sample["nt_seq"][:last_pos]))] + [
            tokens_to_insert[0]["token"]
        ]
        for token in tokens_to_insert[1:]:
            if last_pos != token["pos"]:
                new_seq.append(" ".join(sample["nt_seq"][last_pos : token["pos"]]))
            new_seq.append(token["token"])
            last_pos = token["pos"]

        new_seq.append(" ".join(sample["nt_seq"][last_pos:]))
        
        new_seq = " ".join(new_seq)
        sample["nt_seq"] = new_seq
        
        return sample
    
