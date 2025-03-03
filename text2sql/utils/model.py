import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model


# 모델 ID를 받아 토크나이저 및 모델을 로드 후 파이프라인으로 리턴
def make_inference_pipeline(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # the bitsandbytes library only works on CUDA GPU.
    # https://huggingface.co/docs/bitsandbytes/v0.42.0/installation#installation
    # https://github.com/huggingface/transformers/issues/23970#issuecomment-1598861262
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", load_in_4bit=True,
                                                     bnb_4bit_compute_dtype=torch.float16)
    elif torch.backends.mps.is_available():
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="mps")

        ############################
        # LoRA 적용
        # beomi/Yi-Ko-6B는 LLaMA 기반의 모델임
        # https://huggingface.co/beomi/Yi-Ko-6B/blob/main/config.json
        # 모델 출력해서 선형층(Linear) 확인 후 target_modules 설정
        print(model)
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        # lora_config를 적용해 파라미터를 재구성
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return pipe
