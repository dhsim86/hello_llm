if __name__ == '__main__':
    base_model = 'beomi/Yi-Ko-6B'
    finetuned_model = 'yi-ko-6b-text2sql'

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, PeftModel

    model_name = base_model
    device_map = {"": 0}

    # LoRA와 기초 모델 파라미터 합치기
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    model = PeftModel.from_pretrained(base_model, finetuned_model)
    model = model.merge_and_unload()

    # 토크나이저 설정
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 허깅페이스 허브에 모델 및 토크나이저 저장
    model.push_to_hub(finetuned_model, use_temp_dir=False)
    tokenizer.push_to_hub(finetuned_model, use_temp_dir=False)
