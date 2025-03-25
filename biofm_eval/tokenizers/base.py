import json
import os
from transformers import PreTrainedTokenizer, AddedToken, AutoTokenizer
from typing import Dict, List, Optional, Sequence, Union
from pathlib import Path


ANNOTATION_TOKENS = [
    "[START_CDS]",
    "[END_CDS]",
    "[START_TRANSCRIPT]",
    "[END_TRANSCRIPT]",
    "[START_EXON]",
    "[END_EXON]",
    "[START_CODON]",
]


class AnnotationTokenizer(PreTrainedTokenizer):
    def __init__(
        self,
        characters: Sequence[str],
        variant_characters: Sequence[str],
        model_max_length: int,
        padding_side: str = "left",
        **kwargs,
    ):
        """Annotation + Variant tokenizer for Hugging Face transformers."""
        self._vocab_str_to_int = {
            **{ch: i for i, ch in enumerate(characters)},
            "[CLS]": 6,
            "[SEP]": 7,
            "[UNK]": 8,
            "[PAD]": 9,
            "[EOS]": 10,
            "[BOS]": 11,
            "[SPECIAL]": 12,
            **{ch: i + 13 for i, ch in enumerate(variant_characters)},
        }
        vocab_len = len(self._vocab_str_to_int)
        for i, at in enumerate(ANNOTATION_TOKENS):
            self._vocab_str_to_int[at] = i + vocab_len

        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}

        self.characters = characters
        self.variant_characters = variant_characters
        self.model_max_length = model_max_length
        bos_token = AddedToken("[BOS]", lstrip=False, rstrip=False)
        eos_token = AddedToken("[EOS]", lstrip=False, rstrip=False)
        sep_token = AddedToken("[SEP]", lstrip=False, rstrip=False)
        cls_token = AddedToken("[CLS]", lstrip=False, rstrip=False)
        pad_token = AddedToken("[PAD]", lstrip=False, rstrip=False)
        unk_token = AddedToken("[UNK]", lstrip=False, rstrip=False)

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            unk_token=unk_token,
            add_prefix_space=False,
            model_max_length=model_max_length,
            padding_side=padding_side,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def _tokenize(self, text: str) -> List[str]:
        return text.split()

    def get_vocab(self) -> Dict[str, int]:
        return self._vocab_str_to_int.copy()

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int["[UNK]"])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        result = cls + token_ids_0 + sep
        if token_ids_1 is not None:
            result += token_ids_1 + sep
        return result

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )

        result = ([1] * len(token_ids_0)) + [1]
        if token_ids_1 is not None:
            result += ([1] * len(token_ids_1)) + [1]
        return result

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        result = len(cls + token_ids_0 + sep) * [0]
        if token_ids_1 is not None:
            result += len(token_ids_1 + sep) * [1]
        return result

    def add_tokens(self, new_tokens):
        if isinstance(new_tokens, str):
            new_tokens = [new_tokens]

        tokens_added = 0
        for token in new_tokens:
            if token not in self._vocab_str_to_int:
                new_id = len(self._vocab_str_to_int)
                self._vocab_str_to_int[token] = new_id
                self._vocab_int_to_str[new_id] = token
                tokens_added += 1

        return tokens_added

    def get_config(self) -> Dict:
        return {
            "char_ords": [ord(ch) for ch in self.characters],
            "model_max_length": self.model_max_length,
        }

    @classmethod
    def from_config(cls, config: Dict) -> "AnnotationTokenizer":
        cfg = {}
        cfg["characters"] = [chr(i) for i in config["char_ords"]]
        cfg["model_max_length"] = config["model_max_length"]
        cfg["variant_characters"] = list("ÂṰĜĈ")
        return cls(**cfg)

    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        cfg_file = Path(save_directory) / "tokenizer_config.json"
        cfg = self.get_config()
        with open(cfg_file, "w") as f:
            json.dump(cfg, f, indent=4)

    # @classmethod
    # def from_pretrained(cls, save_directory: Union[str, os.PathLike], **kwargs):
    #     cfg_file = Path(save_directory) / "tokenizer_config.json"
    #     with open(cfg_file) as f:
    #         cfg = json.load(f)
    #     return cls.from_config(cfg)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ):
        """
        Load a tokenizer from either a local directory or a Hugging Face model name.

        Args:
            pretrained_model_name_or_path (str or PathLike): Either a local path to a saved tokenizer
                or a Hugging Face model name (e.g., 'bert-base-uncased')
            **kwargs: Additional arguments to pass to the tokenizer initialization

        Returns:
            AnnotationTokenizer: Initialized tokenizer
        """
        # Check if the input is a local directory with a tokenizer_config.json
        if os.path.isdir(pretrained_model_name_or_path):
            cfg_file = Path(pretrained_model_name_or_path) / "tokenizer_config.json"
            if cfg_file.exists():
                with open(cfg_file) as f:
                    cfg = json.load(f)
                return cls.from_config(cfg)

        # If not a local directory, try loading from Hugging Face
        try:
            # Use AutoTokenizer to load from Hugging Face model name
            hf_tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path, **kwargs
            )

            # Convert Hugging Face tokenizer to your custom tokenizer configuration
            cfg = {
                "char_ords": [ord(c) for c in hf_tokenizer.vocab.keys() if len(c) == 1],
                "model_max_length": hf_tokenizer.model_max_length,
            }

            # Convert using your existing from_config method
            custom_tokenizer = cls.from_config(cfg)

            # Optionally store the original HF tokenizer for reference
            custom_tokenizer._hf_tokenizer = hf_tokenizer

            return custom_tokenizer

        except Exception as e:
            raise ValueError(
                f"Could not load tokenizer from {pretrained_model_name_or_path}: {str(e)}"
            )
