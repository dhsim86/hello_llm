from transformers import AutoTokenizer

if __name__ == '__main__':
    model_id = 'klue/roberta-base'
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print(f"tokenizer: {tokenizer}")
