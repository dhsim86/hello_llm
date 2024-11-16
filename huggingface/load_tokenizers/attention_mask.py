from transformers import AutoTokenizer

if __name__ == '__main__':
    model_id = 'klue/roberta-base'
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    tokenized = tokenizer(["첫 번째 문장은 짧다.", '두 번째 문장은 첫 번째 문장보다 길다.'], padding='longest')
    print(tokenized)
