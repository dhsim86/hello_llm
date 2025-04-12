from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

# GPTQ를 통해 양자화
if __name__ == '__main__':
    model_id = "facebook/opt-125m"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # GPTQConfig를 통해 비트 수 및 사용할 데이터셋, 토크나이저 설정
    # GPTQ에서는 양자화 전후 결과를 최소화하기 위해 데이터 입력 후 결과를 확인하는 과정이
    # 있어 데이터셋 전달해야 한다.
    quantization_config = GPTQConfig(bits=4, dataset="c4", tokenizer=tokenizer)

    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", quantization_config=quantization_config)
    #model = AutoModelForCausalLM.from_pretrained("TheBloke/zephyr-7B-beta-GPTQ",
    #                                             device_map="auto",
    #                                             trust_remote_code=False,
    #                                             revision="main")
