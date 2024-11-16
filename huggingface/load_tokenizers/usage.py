from transformers import AutoTokenizer

if __name__ == '__main__':
    model_id = 'klue/roberta-base'
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    tokenized = tokenizer("토크나이저는 텍스트를 토큰 단위로 나눈다.")
    print(tokenized)

    # 토큰 ID to 토큰
    print(tokenizer.convert_ids_to_tokens(tokenized['input_ids']))

    # 토큰 ID들을 다시 텍스트로 디코드
    print(tokenizer.decode(tokenized['input_ids']))

    # 디코드할 때 특수 토큰 제외
    print(tokenizer.decode(tokenized['input_ids'], skip_special_tokens=True))