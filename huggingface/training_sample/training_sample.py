import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW
import numpy as np


def prepare_dataset():
    from datasets import load_dataset

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

    return train_dataset, test_dataset, valid_dataset


##############################################################################
# 데이터 전처리
def make_dataloader(dataset, batch_size, shuffle=True):
    def tokenize_function(examples):  # 제목(title) 컬럼에 대한 토큰화
        return tokenizer(examples["title"], padding="max_length", truncation=True)

    dataset = dataset.map(tokenize_function, batched=True).with_format("torch")  # 데이터셋에 토큰화(title 컬럼) 수행
    dataset = dataset.rename_column("label", "labels")  # 컬럼 이름 변경
    dataset = dataset.remove_columns(column_names=['title'])  # 불필요한 컬럼(토큰화를 수행한 title 컬럼) 제거

    print(f"after data pre processing: {dataset[0]}")  # 토큰화된 데이터 확인

    # DataLoader 클래스를 사용해 데이터셋을 배치 데이터로 변환
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


##############################################################################
# 학습 수행
def train_epoch(model, data_loader, optimizer):
    # 모델을 학습모드로 변경
    model.train()
    total_loss = 0

    # 배치 데이터를 가져와 모델에 입력으로 전달
    for batch in tqdm(data_loader):
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)  # 모델에 입력할 토큰 아이디
        attention_mask = batch['attention_mask'].to(device)  # 모델에 입력할 어텐션 마스크
        labels = batch['labels'].to(device)  # 모델에 입력할 레이블

        # 모델에 인자로 전달해 모델 계산
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs.loss  # 손실
        loss.backward()  # 손실값 역전파

        optimizer.step()  # 모델 업데이트 (역전파 결과를 바탕으로 모델 업데이트)

        total_loss += loss.item()  # 로스 집계

    avg_loss = total_loss / len(data_loader)
    return avg_loss


def evaluate(model, data_loader):
    # 모델을 추론모드로 설정
    model.eval()

    total_loss = 0
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            logits = outputs.logits  # logits 속성

            loss = outputs.loss
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1)  # 가장 큰 값으로 예측한 카테고리 정보 확인

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = np.mean(np.array(predictions) == np.array(true_labels))  # 실제 정답과 비교하여 정확도 계산
    return avg_loss, accuracy


if __name__ == '__main__':
    train_dataset, test_dataset, valid_dataset = prepare_dataset()
    ##############################################################################
    # 모델 및 토크나이저 로드
    model_id = "klue/roberta-base"
    model = AutoModelForSequenceClassification.from_pretrained(model_id,
                                                               num_labels=len(train_dataset.features['label'].names))
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    ##############################################################################
    # 학습할 디바이스 설정

    # 학습시 NVIDA 그래픽카드 사용을 위한 cuda 사용가능 여부 확인
    print(f"cuda is available: {torch.cuda.is_available()}")

    # 학습시 Mac GPU 사용을 위한 mps 사용가능 여부 확인
    print(f"mps is available: {torch.backends.mps.is_available()}")

    device = "cuda" if torch.cuda.is_available() else \
        "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"using device: {device}")
    device = torch.device(device)

    # 학습을 수행할 디바이스 설정, GPU로 모델 이동
    model.to(device)

    # 데이터로더 만들기
    train_dataloader = make_dataloader(train_dataset, batch_size=8, shuffle=True)
    valid_dataloader = make_dataloader(valid_dataset, batch_size=8, shuffle=False)
    test_dataloader = make_dataloader(test_dataset, batch_size=8, shuffle=False)

    num_epochs = 1
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # 학습 루프, 1-epoch 수행
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        train_loss = train_epoch(model, train_dataloader, optimizer)
        print(f"Training loss: {train_loss}")

        valid_loss, valid_accuracy = evaluate(model, valid_dataloader)
        print(f"Validation loss: {valid_loss}")
        print(f"Validation accuracy: {valid_accuracy}")

    # Testing, 성능 평가
    _, test_accuracy = evaluate(model, test_dataloader)
    print(f"Test accuracy: {test_accuracy}")  # 정확도 0.82

    # 모델의 예측 아이디와 문자열 레이블을 연결할 데이터를 모델 config에 저장
    id2label = {i: label for i, label in enumerate(train_dataset.features['label'].names)}
    label2id = {label: i for i, label in id2label.items()}
    model.config.id2label = id2label
    model.config.label2id = label2id

    # 모델 / 토크나이저를 허깅페이스로 업로드
    # from huggingface_hub import login

    # login(token='')
    # repo_id = f"raveas/roberta-base-klue-ynat-classification"

    # model.push_to_hub(repo_id)
    # tokenizer.push_to_hub(repo_id)
