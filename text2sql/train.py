def prepare_dataset():
    from datasets import load_dataset, Dataset
    from utils.prompt import make_prompt

    # 학습 데이터 다운로드
    train_dataset = load_dataset("shangrilar/ko_text2sql", "origin")["train"]

    train_dataset = train_dataset.to_pandas()
    train_dataset = train_dataset.dropna().sample(frac=1, random_state=42)

    # 평가에 사용할 게임 카테고리(db_id == 1)은 제거
    train_dataset = train_dataset.query("db_id != 1")

    # 다시 Dataset으로 변환
    train_dataset = Dataset.from_pandas(train_dataset)

    # SQL 생성 프롬프트 컬럼 생성
    def make_prompt_request(row):
        row['train'] = make_prompt(row['context'], row['question'], row['answer'])
        return row

    train_dataset = train_dataset.map(make_prompt_request)

    # 필요없는 컬럼 제거
    train_dataset = train_dataset.remove_columns(['db_id', 'context', 'question', 'answer', '__index_level_0__'])
    print(train_dataset[0])

    return train_dataset


def train(model, tokenizer, train_dataset):
    from transformers import Trainer

    ##############################################################################
    # 학습인자와 평가 함수 정의
    training_args = TrainingArguments(
        output_dir='./results',  # 결과를 저장할 폴더
        num_train_epochs=1,  # 학습 에포크 수 (전체 데이터셋 1번만 학습)
        per_device_train_batch_size=8,  # 학습에 사용할 배치 크기
        per_device_eval_batch_size=8,  # evaluation에 사용할 배치 크기
        evaluation_strategy='epoch',  # 평가를 수행할 빈도, 1-epoch 학습이 끝날 때 평가를 수행
        learning_rate=5e-5,  # 학습률 지정
        push_to_hub=False)  # 모델 학습 후 huggingface에 업로드 여부

    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=train_dataset,
                      tokenizer=tokenizer)

    trainer.train()
    trainer.model.save_pretrained('yi-ko-6b-text2sql');


if __name__ == '__main__':
    from transformers import TrainingArguments
    from utils.model import load_model_and_tokenizer

    # 모델 로드
    model, tokenizer = load_model_and_tokenizer('beomi/Yi-Ko-6B')

    # 학습 데이터셋 준비
    train_dataset = prepare_dataset()
    train(model, tokenizer, train_dataset)
