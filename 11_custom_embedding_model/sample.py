import json

################################################
from sentence_transformers import SentenceTransformer, models

# klue/raberta-base 모델을 로드
transformer_model = models.Transformer('klue/roberta-base')
# 임베딩 모델을 위한 풀링 레이어 추가
pooling_layer = models.Pooling(transformer_model.get_word_embedding_dimension(), pooling_mode_mean_tokens=True)

# 모델과 풀링 레이어를 결합하여 SentenceTransformer 모델 생성
# 추가 학습된 상태가 아니므로, 이 임베딩 모델은
# 언어 모델의 출력을 단순히 평균 내 고정된 차원의 벡터로 생성
embedding_model = SentenceTransformer(modules=[transformer_model, pooling_layer])

################################################
from datasets import load_dataset

# KLUE STS 데이터셋을 로드
# STS 데이터셋은 2개의 문장이 서로 얼마나 유사한지 점수를 매긴 데이터셋
# 학습 데이터셋과 평가 데이터셋을 나누어 로드
klue_sts_train = load_dataset("klue", "sts", split="train")
klue_sts_test = load_dataset("klue", "sts", split="validation")

print(json.dumps(klue_sts_train[0], ensure_ascii=False))

################################################
# 학습 데이터셋의 10%를 검증 데이터셋으로 분리
klue_sts_train = klue_sts_train.train_test_split(test_size=0.1, seed=42)
klue_sts_train, klue_sts_eval = klue_sts_train['train'], klue_sts_train['test']

from sentence_transformers import InputExample


################################################
# 유사도 점수를 0~1로 정규화 후, InputExample 객체로 변환
# InputExample: SentenceTransformer에서 데이터를 관리하는 형식
def prepare_sts_examples(dataset):
    examples = []
    for data in dataset:
        # 입력 문장 쌍을 리스트 형태로 입력
        # label은 5로 나누어 0~1 사이의 값으로 정규화
        examples.append(InputExample(texts=[data['sentence1'], data['sentence2']],
                                     label=data['labels']['label'] / 5.0))
    return examples


train_examples = prepare_sts_examples(klue_sts_train)
eval_examples = prepare_sts_examples(klue_sts_eval)
test_examples = prepare_sts_examples(klue_sts_test)

################################################
# 학습을 위한 배치 데이터 생성
from torch.utils.data import DataLoader

# 학습 데이터셋을 DataLoader에 넣고 배치 크기가 16인 배치 데이터 생성
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

################################################
# 검증 데이터셋과 평가 데이터셋을 EmbeddingSimilarityEvaluator에 사용해
# 임베딩 모델의 성능을 평가할 때 사용하도록 준비
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

# 평가 객체 생성
eval_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(eval_examples)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_examples)

################################################
# 언어 모델을 그대로 문장 임베딩 모델로 만든 embedding_model의 성능을 평가
# 얼마나 문장의 의미를 잘 담아 문장 임베딩을 생성하는지 확인

# 평가 객체에 embedding_model을 넣고 테스트
print(test_evaluator(embedding_model))

################################################
# 임베딩 모델 학습
from sentence_transformers import losses

# 4 에폭 동안 학습
num_epochs = 4
model_name = 'klue/roberta-base'
model_save_path = 'output/training_sts_' + model_name.replace("/", "-")

# 손실함수로 CosineSimilarityLoss 사용
# 학습 데이터를 문장 임베딩으로 변환 후,
# 두 문장 사이의 코사인 유사도와 정답 유사도를 비교해 학습
train_loss = losses.CosineSimilarityLoss(model=embedding_model)

# 임베딩 모델 학습
# 학습 데이터 및 손실함수, 검증에 사용할 평가 객체를 인자로 전달하여 학습
embedding_model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=eval_evaluator,  # 평가 객체 (검증 데이터)
    epochs=num_epochs,
    evaluation_steps=1000,
    warmup_steps=100,
    output_path=model_save_path
)

################################################
# 학습한 임베딩 모델의 성능을 확인
trained_embedding_model = SentenceTransformer(model_save_path)
test_evaluator(trained_embedding_model)
