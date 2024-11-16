from transformers import AutoTokenizer

if __name__ == '__main__':
    model_id = 'klue/roberta-base'
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    tokenized = tokenizer(["첫 번째 문장", "두 번째 문장"])
    print(tokenized)

    decoded = tokenizer.batch_decode(tokenized['input_ids'])
    print(decoded)
