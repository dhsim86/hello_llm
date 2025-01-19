from transformer_examples.tokens import embedding_text

from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F


# 쿼리, 키, 값 벡터 생성
def get_qkv(input_embedding: str):
    embedding_dim = 16
    head_dim = 16

    # nn.Linear 인스턴스는 서로 독립적, 서로 다른 가중치를 가지고 학습
    # Linear 레이어는 호출시 내부 가중치를 적용 후 변환됨 => 동일 입력, 다른 출력
    # 쿼리, 키, 값을 변환하기 위한 가중치
    weight_q = nn.Linear(embedding_dim, head_dim) # nn.Linear는 선형 변환을 수행하는 모듈 (입력 / 출력 텐서의 크기가 16)
    weight_k = nn.Linear(embedding_dim, head_dim)
    weight_v = nn.Linear(embedding_dim, head_dim)

    ##################################################################
    # 2. 임베딩된 각 쿼리 및 키, 그리고 값을 가중치(Wq / Wk / Wv)를 통해 변환
    queries = weight_q(input_embedding) # [1, 5, 16]
    keys = weight_k(input_embedding)    # [1, 5, 16]
    values = weight_v(input_embedding)  # [1, 5, 16]

    print(f"queries: {queries}")
    print(f"keys: {keys}")
    print(f"values: {values}")

    return queries, keys, values


def compute_attention(queries, keys, values, is_causal=False):
    print("---------------compute_attention-----------------")
    dim_k = queries.size(-1)    # 16, 행렬의 디멘션(차원) 크기를 구하기

    ##################################################################
    # 3. 변환된 쿼리와 키의 관련도를 계산하여 스코어를 산출 (스케일 점곱)
    #    - 변환된 쿼리와 키 벡터를 내적한 후, 키 벡터의 차원의 제곱근으로 나누어 준다.
    #    - 쿼리와 키를 곱하고, 분산이 커지는 것을 방지하기 위해 임베딩 차원 수의 제곱근으로 나눈다.
    # transpose: 차원을 맞교환, 교환하고자 하는 인덱스를 파라미터로 넘김
    # keys.shape => [1, 5, 16]
    # keys.transpose(-2, -1).shape => [1, 16, 5]
    # queries @ key.transpose(-2, -1) => [1, 5, 16] * [1, 16, 5] => [1, 5, 5]   (입력 문장은 5개의 토큰으로 구성)
    print(f"keys.shape: {keys.shape}")
    print(f"keys.transpose(-2, -1).shape: {keys.transpose(-2, -1).shape}")
    scores = queries @ keys.transpose(-2, -1) / sqrt(dim_k)

    # 마스크 여부
    if is_causal:
        # TODO: 버그 수정
        query_length = queries.size(1) # 디코더에서 실행시 2로 변경
        key_length = keys.size(1) # 디코더에서 실행시 2로 변경
        temp_mask = torch.ones(query_length, key_length, dtype=torch.bool).tril(diagonal=0)
        scores = scores.masked_fill(temp_mask == False, float("-inf"))

    print(f"scores: {scores}")

    ##################################################################
    # 4. 스코어를 softmax 취하여 가중치(score)로 계산
    #    - 쿼리와 키를 곱해 계산한 score를 합이 1이 되도록 소프트맥스를 취해 가중치로 바꾼다.
    weight = F.softmax(scores, dim=-1)
    print(f"weight: {weight}")

    ###################################################################
    # 5. Wv를 통해 변환된 값을 가중치(score)와 곱해 입력과 동일한 형태의 출력 벡터 생성
    #    - 가중치와 값을 곱해 입력과 동일한 형태의 출력
    #      [1, 5, 5] * [1, 5, 16] => [1, 5, 16]
    return weight @ values


if __name__ == '__main__':
    ##################################################################
    # 1. 임베딩 벡터를 가져온다.
    queries, keys, values = get_qkv(embedding_text.get_input_embeddings())
    result = compute_attention(queries, keys, values)
    print(f"result: {result}")
    print(f"result.shape: {result.shape}")

    masked_result = compute_attention(queries, keys, values, True)
    print(f"masked_result: {masked_result}")
    print(f"masked_result.shape: {masked_result.shape}")
