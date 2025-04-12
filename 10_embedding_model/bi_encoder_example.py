# Sentence-Transformers 라이브러리를 통한 바이인코더 사용
from sentence_transformers import SentenceTransformer, models

# 사용할 BERT 모델, models 모듈의 Transformer 클래스를 통해 허깅페이스의 모델을 로드
word_embedding_model = models.Transformer('klue/roberta-base')

# models 모듈의 Pooling 클래스를 통해 풀링 층을 생성
# 풀링 층에 입력으로 들어오는 토큰 임베딩의 차원을
# get_word_embedding_dimension 메서드로 설정
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

# 리스트 형태로 넘겨 바이인코더 생성
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

print(f"모델 구성: {model}")


def mean_pooling(model_output, attention_mask):
    import torch

    """
    mean pooling을 통해 문장 임베딩을 생성하는 함수
    :param model_output: 모델의 출력값
    :param attention_mask: 어텐션 마스크, 패딩 토큰의 위치를 확인할 수 있다.
    :return: 문장 임베딩
    """
    token_embeddings = model_output[0]  # 언어 모델의 출력 중 마지막 층의 출력만 사용

    # 입력이 패딩 토큰인 부분은 평균 계산에서 무시하기 위해 input_mask_expanded를 생성 후 출력 임베딩에 곱한다.
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

    # 출력 임베딩의 합을 실제 토큰의 입력 수로 나눈다.
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)  # 최소값 설정
    sentence_embeddings = sum_embeddings / sum_mask  # 문장 임베딩 계산

    return sentence_embeddings


def max_pooling(model_output, attention_mask):
    import torch
    """
    max pooling을 통해 문장 임베딩을 생성하는 함수
    :param model_output: 모델의 출력값
    :param attention_mask: 어텐션 마스크, 패딩 토큰의 위치를 확인할 수 있다.
    :return: 문장 임베딩
    """
    token_embeddings = model_output[0]  # 언어 모델의 출력 중 마지막 층의 출력만 사용

    # 입력이 패딩 토큰인 부분은 평균 계산에서 무시하기 위해 input_mask_expanded를 생성 후 출력 임베딩에 곱한다.
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    # 패딩 토큰인 부분에 -1e9로 아주 작은 값을 입력해 최대값이 될 수 없도록 설정
    token_embeddings[input_mask_expanded == 0] = -1e9
    # 출력 임베딩의 토큰 길이 차원에서 가장 큰 값을 찾는다.
    max_embeddings = torch.max(token_embeddings, 1)[0]

    return max_embeddings
