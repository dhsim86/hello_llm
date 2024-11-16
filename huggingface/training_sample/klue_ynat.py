from datasets import load_dataset

if __name__ == '__main__':
    klue_tc_train = load_dataset('klue', 'ynat', split='train')
    klue_tc_eval = load_dataset('klue', 'ynat', split='validation')

    ##############################################################################
    # 데이터셋 확인
    print(klue_tc_train)
    # 첫 번째 데이터 확인
    print(klue_tc_train[0])
    # label 컬럼의 항목별 이름 확인
    print(klue_tc_train.features['label'].names)

    ##############################################################################
    # 학습에 필요없는 컬럼 제거
    # guid / url / date 컬럼 제거
    klue_tc_train = klue_tc_train.remove_columns(['guid', 'url', 'date'])
    klue_tc_eval = klue_tc_eval.remove_columns(['guid', 'url', 'date'])
    print(klue_tc_train)

    ##############################################################################
    # 카테고리를 확인할 수 있는 label_str 컬럼 추가
    # label 확인
    print(klue_tc_train.features['label'])
    # ClassLabel, 레이블 ID와 카테고리를 연결하는 객체
    klue_tc_label = klue_tc_train.features['label']


    def make_str_label(batch):
        # 레이블 ID를 카테고리로 변환하는 int2str 메서드를 통해
        # 레이블 ID를 카테고리 이름으로 변환
        batch['label_str'] = klue_tc_label.int2str(batch['label'])
        return batch


    # label_str 컬럼 추가
    klue_tc_train = klue_tc_train.map(make_str_label, batched=True)
    print(klue_tc_train[0])

    # 학습용과 검증용, 테스트용 데이터셋으로 분리
    train_dataset = klue_tc_train.train_test_split(test_size=10000, shuffle=True, seed=42)['test']

    dataset = klue_tc_eval.train_test_split(test_size=1000, shuffle=True, seed=42)
    test_dataset = dataset['test']
    valid_dataset = dataset['train'].train_test_split(test_size=1000, shuffle=True, seed=42)['test']

    ##############################################################################
    # 트레이너 API를 활용한 학습을 수행
    # 학습 준비
    import torch.cuda
    import torch
    import numpy as np
    from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer

    # 모델과 토크나이저를 로드, 데이터셋에 토큰화를 수행
    model_id = 'klue/roberta-base'

    # AutoModelForSequenceClassification 클래스로 모델을 불러오면
    # 분류 헤드 부분은 랜덤으로 초기화되어 있다.
    # 분류 헤드 부분의 분류 클래스 수(num_labels)를 데이터셋의 레이블 수로 지정
    model = AutoModelForSequenceClassification.from_pretrained(model_id,
                                                               num_labels=len(train_dataset.features['label'].names))
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # title 컬럼에 토큰화를 수행
    def tokenize_function(examples):
        return tokenizer(examples['title'], padding='max_length', truncation=True)
    # train / valid / test 데이터셋의 title 컬럼에 토큰화를 수행
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    valid_dataset = valid_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    ##############################################################################
    # 학습인자와 평가 함수 정의
    training_args = TrainingArguments(
        output_dir='./results', # 결과를 저장할 폴더
        num_train_epochs=1,     # 학습 에포크 수 (전체 데이터셋 1번만 학습)
        per_device_train_batch_size=8,  # 학습에 사용할 배치 크기
        per_device_eval_batch_size=8,   # evaluation에 사용할 배치 크기
        evaluation_strategy='epoch',    # 평가를 수행할 빈도, 1-epoch 학습이 끝날 때 평가를 수행
        learning_rate=5e-5,             # 학습률 지정
        push_to_hub=False)              # 모델 학습 후 huggingface에 업로드 여부

    # 평가 지표 정의 (accuracy)
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {"accuracy": (predictions == labels).mean()}

    ##############################################################################
    # Trainier를 이용한 학습
    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=train_dataset,
                      eval_dataset=valid_dataset,
                      tokenizer=tokenizer,
                      compute_metrics=compute_metrics)

    # 학습시 NVIDA 그래픽카드 사용을 위한 cuda 사용가능 여부 확인
    print(f"cuda is available: {torch.cuda.is_available()}")
    # 학습
    trainer.train()

    # 테스트 및 평가
    result = trainer.evaluate(test_dataset)

    # 평가 결과 출력
    print(result)
