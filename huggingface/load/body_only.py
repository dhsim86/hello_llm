from transformers import AutoModel, AutoTokenizer

if __name__ == '__main__':
    model_id = 'klue/roberta-base'
    model = AutoModel.from_pretrained(model_id)

    print(f"model: {model}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    encoded_input = tokenizer('LLM을 활용한 실전 애플리케이션 개발', return_tensors='pt')
    output = model(**encoded_input)

    print(f"output: {output}")
