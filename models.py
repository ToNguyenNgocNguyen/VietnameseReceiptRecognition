import torch
from transformers import (AutoModel, AutoTokenizer,
                          AutoModelForCausalLM, AutoProcessor)


class VLLMModel:
    def __init__(self, model_name: str, device: str, dtype: torch.dtype):
        # Load the model and tokenizer
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).eval().to(device)  # Move model to GPU (if available)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


    @classmethod
    def from_pretrained(cls, model_name: str, device: str, dtype: torch.dtype):
        # Create an instance and return model, tokenizer
        instance = cls(model_name, device, dtype)
        return instance.model, instance.tokenizer
    

class VLLMModelForCausalLM:
    def __init__(self, model_name: str, device: str, dtype: torch.dtype):
        # Load the model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).eval().to(device)

        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    @classmethod
    def from_pretrained(cls, model_name: str, device: str, dtype: torch.dtype):
        # Create an instance and return model, processor
        instance = cls(model_name, device, dtype)
        return instance.model, instance.processor
