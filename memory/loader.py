from transformers import AutoModelForCausalLM, AutoTokenizer

from check_gpu_memory import print_gpu_utilization


# 사용할 모델의 ID와 어떤 방식을 사용할지 입력받는 peft(PEFT 튜닝) 정의
def load_model_and_tokenizer(model_id, peft=None):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if peft is None:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map={"": 0})

    print_gpu_utilization()
    return model, tokenizer


# GPU 메모리 사용량: 2.575 GB
# 모델 파라미터 타입: torch.float16
if __name__ == '__main__':
    model_id = "EleutherAI/polyglot-ko-1.3b"
    model, tokenizer = load_model_and_tokenizer(model_id)
    print(f"모델 파라미터 타입: {model.dtype}")
