import torch
from transformers import MistralForCausalLM, AutoConfig
from typing import Dict, Any


class AnnotatedModel(MistralForCausalLM):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
    

