# https://github.com/huggingface/transformers/issues/23970#issuecomment-1934826430

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "meta-llama/Llama-2-7b-hf"

bnb_config = BitsAndBytesConfig(
    #load_in_4bit=True,
    load_in_4bit=False,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

print(torch.backends.mps.is_available())

model = AutoModelForCausalLM.from_pretrained(
    model_id, quantization_config=bnb_config, trust_remote_code=True, device_map='mps'
)