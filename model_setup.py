# src/model_setup.py

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# --- Configuration ---
BASE_MODEL_NAME = "bigscience/bloom-3b"

def load_model_and_tokenizer():
    """
    Loads the quantized base model and tokenizer.
    """
    # Configuration for 8-bit quantization
    quantization_config_8bit = BitsAndBytesConfig(
        load_in_8bit=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=quantization_config_8bit,
        device_map='auto',
    )

    tokenizer = AutoTokenizer.from_pretrained("bigscience/tokenizer")
    return model, tokenizer

class CastOutputToFloat(nn.Sequential):
  """
  A custom layer to cast the output of a model's head to float32 for stability.
  """
  def forward(self, x):
    return super().forward(x).to(torch.float32)

def prepare_model_for_lora(model):
    """
    Prepares the model for LoRA training by freezing its parameters and
    handling output casting.
    """
    # Freeze all model parameters
    for param in model.parameters():
        param.requires_grad = False
        # Cast small parameters (like layernorm) to fp32 for stability
        if param.ndim == 1:
            param.data = param.data.to(torch.float32)

    # Reduce the number of stored activations
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # Cast the output of the language model head to float32
    model.lm_head = CastOutputToFloat(model.lm_head)
    return model
