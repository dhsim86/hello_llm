from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

from check_gpu_memory import print_gpu_utilization


# 사용할 모델의 ID와 어떤 방식을 사용할지 입력받는 peft(PEFT 튜닝) 정의
def load_model_and_tokenizer(model_id, peft=None):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if peft is None:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map={"": 0})
    elif peft == 'lora':
        # peft 라이브러리를 통해 LoRA를 적용
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map={"": 0})
        
        # peft의 LoraConfig를 통해 LoRA의 하이퍼파라미터 설정
        # r 및 alpha, 그리고 LoRA를 어떤 가중치에 적용할지 결정하는 target_modules 설정
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["query_key_value"], # query, key, value 가중치에 적용
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        # lora_config를 적용해 파라미터를 재구성
        model = get_peft_model(model, lora_config)
        
        # 모델 재구성 후 학습 파라미터의 수와 비중을 확인
        model.print_trainable_parameters()

    print_gpu_utilization()
    return model, tokenizer


# GPU 메모리 사용량: 2.575 GB
# 모델 파라미터 타입: torch.float16
if __name__ == '__main__':
    model_id = "EleutherAI/polyglot-ko-1.3b"
    model, tokenizer = load_model_and_tokenizer(model_id)
    print(f"모델 파라미터 타입: {model.dtype}")
