from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from memory.utils import print_gpu_utilization, clean_gpu_memory

if __name__ == '__main__':
    clean_gpu_memory()

    print("before")
    print_gpu_utilization()

    # 8비트 양자화 모델 불러오기
    bnb_config_8bit = BitsAndBytesConfig(load_in_8bit=True)
    model_8bit = AutoModelForCausalLM.from_pretrained("facebook/opt-350m",
                                                      quantization_config=bnb_config_8bit)
    print("model_8bit")
    print_gpu_utilization()

    clean_gpu_memory()
    # 4비트 양자화 모델 불러오기
    bnb_config_4bit = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
    model_4bit = AutoModelForCausalLM.from_pretrained("facebook/opt-350m",
                                                      low_cpu_mem_usage=True,
                                                      quantization_config=bnb_config_4bit)
    print("model_4bit")
    print_gpu_utilization()
