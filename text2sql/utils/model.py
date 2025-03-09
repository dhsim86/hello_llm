import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM


# 모델 ID를 받아 토크나이저 및 모델을 로드 후 파이프라인으로 리턴
def make_inference_pipeline(model_id):
    model, tokenizer = load_model_and_tokenizer(model_id)

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return pipe


def load_model_and_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    ###################################################################
    # NVIDIA GPU
    # the bitsandbytes library only works on CUDA GPU.
    # https://huggingface.co/docs/bitsandbytes/v0.42.0/installation#installation
    # https://github.com/huggingface/transformers/issues/23970#issuecomment-1598861262
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", load_in_4bit=True,
                                                     bnb_4bit_compute_dtype=torch.float16)
    ###################################################################
    # macOS with MPS
    elif torch.backends.mps.is_available():
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)

    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

    return model, tokenizer
