from transformer.tokens import embedding_text

from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F


# 쿼리, 키, 값 벡터 생성
def get_qkv(input_embedding: str):
    embedding_dim = 16
    head_dim = 16

    # 쿼리, 키, 값을 계산하기 위한 변환
    weight_q = nn.Linear(embedding_dim, head_dim)
    weight_k = nn.Linear(embedding_dim, head_dim)
    weight_v = nn.Linear(embedding_dim, head_dim)

    # 변환수행
    queries = weight_q(input_embedding)
    keys = weight_k(input_embedding)
    values = weight_v(input_embedding)

    print(f"queries: {queries}")
    print(f"keys: {keys}")
    print(f"values: {values}")

    return queries, keys, values


def compute_attention(queries, keys, values, is_causal=False):
    print("---------------compute_attention-----------------")
    dim_k = queries.size(-1)

    # 쿼리와 키를 곱하고, 분산이 커지는 것을 방지하기 위해 임베딩 차원 수의 제곱근으로 나눈다.
    scores = queries @ keys.transpose(-2, -1) / sqrt(dim_k)

    # 마스크 여부
    if is_causal:
        query_length = queries.size(2)
        key_length = keys.size(2)
        temp_mask = torch.ones(query_length, key_length, dtype=torch.bool).tril(diagonal=0)
        scores = scores.masked_fill(temp_mask == False, float("-inf"))

    print(f"scores: {scores}")

    # 쿼리와 키를 곱해 계산한 score를 합이 1이 되도록 소프트맥스를 취해 가중치로 바꾼다.
    weight = F.softmax(scores, dim=-1)
    print(f"weight: {scores}")

    # 가중치와 값을 곱해 입력과 동일한 형태의 출력
    return weight @ values


if __name__ == '__main__':
    queries, keys, values = get_qkv(embedding_text.get_input_embeddings())
    result = compute_attention(queries, keys, values)
    masked_result = compute_attention(queries, keys, values, True)

    print(f"result: {result}")
    print(f"masked_result: {result}")
