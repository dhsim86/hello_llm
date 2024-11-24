from datasets import load_dataset
from transformers import pipeline

if __name__ == '__main__':
    dataset = load_dataset('klue', 'ynat', split='validation')

    model_id = 'raveas/roberta-base-klue-ynat-classification'

    model_pipeline = pipeline('text-classification', model=model_id)

    result = model_pipeline(dataset['title'][:5])

    # 예측 확률이 가장 높은 레이블과 확률을 리턴
    print(f"result: {result}")
