import torch
from datasets import load_dataset
from torch.nn.functional import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class CustomPipeline:
    def __init__(self, model_id):
        # 모델 ID에 맞는 모델 및 토크나이저 로드
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model.eval()

    def __call__(self, texts):
        # 1. 토큰화 수행
        tokenized = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

        # 2. 모델 추론
        with torch.no_grad():
            outputs = self.model(**tokenized)
            logits = outputs.logits

        # 3. 가장 큰 예측 확률을 가지는 클래스를 추출 후 결과를 반환
        probabilities = softmax(logits, dim=-1)
        scores, labels = torch.max(probabilities, dim=-1)
        label_str = [self.model.config.id2label[label_idx] for label_idx in labels.tolist()]

        return [{"label": label, "score": score.item()}
                for label, score in zip(label_str, scores)]


if __name__ == '__main__':
    dataset = load_dataset('klue', 'ynat', split='validation')

    custom_pipeline = CustomPipeline('raveas/roberta-base-klue-ynat-classification')
    result = custom_pipeline(dataset['title'][:10])

    # 예측 확률이 가장 높은 레이블과 확률을 리턴
    print(f"input: {dataset['title'][:10]}")
    print(f"result: {result}")
