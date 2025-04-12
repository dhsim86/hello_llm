from transformers import AutoTokenizer

if __name__ == '__main__':
    bert_tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')
    print(bert_tokenizer([['첫 번째 문장', '두 번째 문장']]))

    roberta_tokenizer = AutoTokenizer.from_pretrained('klue/roberta-base')
    print(roberta_tokenizer([['첫 번째 문장', '두 번째 문장']]))

    en_roberta_tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    print(en_roberta_tokenizer([['first sentence', 'second sentence']]))
