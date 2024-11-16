from transformers import AutoModel, AutoTokenizer

if __name__ == '__main__':
    text = "what is Huggingface Transformers?"

    # BERT
    bert_model = AutoModel.from_pretrained('bert-base-uncased')  # 모델 불러오기
    bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')  # 토크나이저 불러오기
    encoded_input = bert_tokenizer(text, return_tensors='pt')  # 입력 토큰화
    bert_output = bert_model(**encoded_input)  # 모델에 입력
    print(f"[BERT] encoded_input: {encoded_input}")
    print(f"[BERT] output: {bert_output}")

    # GPT-2 모델
    gpt_model = AutoModel.from_pretrained('gpt2')  # 모델 불러오기
    gpt_tokenizer = AutoTokenizer.from_pretrained('gpt2')  # 토크나이저 불러오기
    encoded_input = gpt_tokenizer(text, return_tensors='pt')  # 입력 토큰화
    gpt_output = gpt_model(**encoded_input)  # 모델에 입력
    print(f"[GPT-2] encoded_input: {encoded_input}")
    print(f"[GPT-2] output: {gpt_output}")
