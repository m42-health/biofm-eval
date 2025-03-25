import torch
from typing import List, Tuple, Dict
from transformers import AutoTokenizer
from datasets import DatasetDict, Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression

from biofm_eval.models.base import AnnotatedModel
from biofm_eval.annotators.base import Annotator


def extract_decoder_variant_hidden_states(
    hidden_states: torch.Tensor, input_ids: torch.Tensor, pad_token_id: int
):
    mask = input_ids != pad_token_id
    # Get the indices of the last True value in each sequence
    last_non_padding_idx = mask.sum(dim=1) - 2
    last_hidden_states = hidden_states[
        torch.arange(hidden_states.shape[0]), last_non_padding_idx
    ]
    return last_hidden_states


@dataclass
class VariantEmbedding:
    ref: torch.Tensor
    alt: torch.Tensor
    ref_right: torch.Tensor
    alt_right: torch.Tensor


def aggregate(embeddings: List[VariantEmbedding]) -> VariantEmbedding:
    ref, alt, ref_right, alt_right = map(
        np.concatenate,
        [
            [e.ref for e in embeddings],
            [e.alt for e in embeddings],
            [e.ref_right for e in embeddings],
            [e.alt_right for e in embeddings],
        ],
    )
    return VariantEmbedding(ref, alt, ref_right, alt_right)


class Embedder:
    def __init__(self, model: AnnotatedModel, tokenizer: AutoTokenizer):
        self.model = model
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.tokenizer = tokenizer

    def get_sequence_embeddings(self, batch: List[str]) -> torch.Tensor:
        def preprocess_seq(strings: List[str]):
            return [" ".join(s) for s in strings]

        batch = preprocess_seq(batch)

        return self._get_sequence_embeddings(batch)

    def _get_sequence_embeddings(self, batch: List[str]) -> torch.Tensor:
        tok_seq = self.tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True
        )
        # Check if CUDA is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tok_seq = {
            k: v.to(device)
            for k, v in tok_seq.items()
            if k not in ["token_type_ids", "attention_mask"]
        }

        output = self.model(**tok_seq, output_hidden_states=True)
        embeddings = extract_decoder_variant_hidden_states(
            output.hidden_states[-1], tok_seq["input_ids"], self.tokenizer.pad_token_id
        )
        return embeddings.detach().float().cpu()

    def get_variant_embeddings(self, batch: Dict[str, List[str]]) -> VariantEmbedding:
        ref_embeddings = self._get_sequence_embeddings(batch["ref_left"])
        alt_embeddings = self._get_sequence_embeddings(batch["alt_left"])
        ref_right_embeddings = self._get_sequence_embeddings(batch["ref_right"])
        alt_right_embeddings = self._get_sequence_embeddings(batch["alt_right"])
        return VariantEmbedding(
            ref_embeddings, alt_embeddings, ref_right_embeddings, alt_right_embeddings
        )

    def get_dataset_embeddings(
        self, dataset: Dataset, batch_size: int = 8
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get embeddings for a dataset
        Each dataset item should have the following fields:
            - ref_left: str
            - alt_left: str
            - ref_right: str
            - alt_right: str
            - label: int (for a supervised dataset)
        Use Annotator class to encode variants and annotate the dataset
        Args:
            dataset: Dataset of DNA sequences to get embeddings
            batch_size: Batch size

        Returns:
            A dictionary of embeddings and labels
        """
        loader = DataLoader(
            dataset,
            num_workers=4,
            shuffle=False,
            drop_last=False,
            prefetch_factor=2,
            batch_size=batch_size,
        )
        labels = []
        embeddings = []
        with torch.inference_mode(True):
            for i, batch in tqdm(enumerate(loader)):
                variant_embedding = self.get_variant_embeddings(batch)
                embeddings.append(variant_embedding)
                if "label" in batch.keys():
                    labels.extend(batch["label"])

            aggregated_embedding = aggregate(embeddings)
            x = np.concatenate(
                [
                    (aggregated_embedding.ref + aggregated_embedding.ref_right) / 2,
                    (aggregated_embedding.alt + aggregated_embedding.alt_right) / 2,
                ],
                axis=1,
            )
            y = np.array(labels)

        return {"embeddings": x, "labels": y}

    def linear_probing(
        self, dataset_dict: DatasetDict, batch_size: int = 8
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Train a linear model on the embeddings
        Args:
            dataset_dict: DatasetDict to train on, should have 'train' and 'test' keys
            batch_size: Batch size

        Returns:
            Tuple of numpy arrays of predicted labels and predicted probabilities for the test set
        """
        train_data = self.get_dataset_embeddings(dataset_dict["train"], batch_size)
        x, y = train_data["embeddings"], train_data["labels"]

        test_data = self.get_dataset_embeddings(dataset_dict["test"], batch_size)
        x_test, y_test = test_data["embeddings"], test_data["labels"]

        model = LogisticRegression(max_iter=5000)
        model.fit(x, y)
        y_test_pred = model.predict(x_test)
        y_test_pred_proba = model.predict_proba(x_test)

        return y_test_pred, y_test_pred_proba[:, 1]
