from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# AWQ를 통해 양자화
if __name__ == '__main__':
    model_name_or_path = "TheBloke/zephyr-7B-beta-AWQ"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=False)
    model = AutoAWQForCausalLM.from_pretrained(model_name_or_path, fuse_layers=True,
                                               trust_remote_code=False, safetensors=True)
