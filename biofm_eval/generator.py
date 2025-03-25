import torch
from typing import List, Tuple, Dict
from transformers import AutoTokenizer
from biofm_eval.models.base import AnnotatedModel


class Generator:
    def __init__(self, model: AnnotatedModel, tokenizer: AutoTokenizer):
        self.model = model
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.tokenizer = tokenizer

    def tokenize_sequence(self, batch: List[str]) -> torch.Tensor:
        def preprocess_seq(strings: List[str]):
            return [" ".join(s) for s in strings]

        batch = preprocess_seq(batch)

        tok_seq = self.tokenizer(batch, return_tensors="pt", padding=True)
        # Check if CUDA is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tok_seq = {
            k: v.to(device)
            for k, v in tok_seq.items()
            if k not in ["token_type_ids", "attention_mask"]
        }
        return tok_seq

    def generate(self, batch: List[str], **kwargs) -> List[str]:
        # Tokenize input sequences
        model_inputs = self.tokenize_sequence(batch)

        generated_ids = self.model.generate(**model_inputs, **kwargs)

        output_seq = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )

        return output_seq
