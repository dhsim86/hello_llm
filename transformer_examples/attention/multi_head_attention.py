import torch.nn as nn

from transformer_examples.attention.attention_test import compute_attention
from transformer_examples.tokens import embedding_text

##################################################################
# 멀티 헤드 어텐션을 위한 하이퍼파라미터
# token_embed_dim: 임베딩 벡터의 크기
# d_model: 모델의 차원 크기, Q, K, V 벡터를 생성하는데 사용되는 가중치의 크기
# n_head: 헤드의 갯수
class MultiHeadAttention(nn.Module):
    def __init__(self, token_embed_dim, d_model, n_head, is_causal=False):
        super().__init__()
        self.n_head = n_head
        self.is_causal = is_causal

        self.weight_q = nn.Linear(token_embed_dim, d_model)
        self.weight_k = nn.Linear(token_embed_dim, d_model)
        self.weight_v = nn.Linear(token_embed_dim, d_model)

        self.concat_linear = nn.Linear(d_model, d_model)

    def forward(self, queries, keys, values):
        B, T, C = queries.size()
        # B: 배치 크기
        # T: 시퀀스 길이 (각 문장의 단어 갯수)
        # C: 임베딩 차원: 16
        print(f"B: {B}, T: {T}, C: {C}")

        ##################################################################
        # 1. 쿼리와 키, 값을 n_head로 쪼갠다. (Q, K, V 벡터가 처음 통과하는 여러 선형 층에 대응)
        #   - [1, 5, 16] => [1, 5, 4, 4] => [1, 4, 5, 4]
        #     - [1, 5, 16] 형태를 [1, 5, 4, 4]로 변환 후 transpose 하면 [1, 4, 5, 4]
        #   - 배치, 헤드, 시퀀스 길이, 헤드 차원 순서가 되어 각 헤드별로 독립적인 어텐션 계산이 가능
        queries = self.weight_q(queries).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        keys = self.weight_k(keys).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        values = self.weight_v(values).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        ##################################################################
        # 2. 각각의 어텐션을 계산 (head 별로 어텐션 계산)
        #  -  [1, 4, 5, 4] * [1, 4, 4, 5] => [1, 4, 5, 5] => [1, 4, 5, 5] * [1, 4, 5, 4] => [1, 4, 5, 4]
        attention = compute_attention(queries, keys, values, self.is_causal)

        ##################################################################
        # 3. 입력과 같은 형태로 다시 변환 (어텐션 결과를 다시 연결하는 단계)
        #   - [1, 4, 5, 4] => [1, 5, 4, 4]가 되고, view(B, T, C) 를 실행하면 [1, 5, 4*4]
        output = attention.transpose(1, 2).contiguous().view(B, T, C)

        ##################################################################
        # 4. 마지막 선형 층을 통과시키고 최종 결과를 반환 (마지막 선형 층)
        #   - ([1, 5, 4*4] => [1, 5, d_model 차원])
        output = self.concat_linear(output)

        return output


if __name__ == '__main__':
    n_head = 4
    input_embeddings = embedding_text.get_input_embeddings()

    mh_attention = MultiHeadAttention(16, 16, n_head)
    after_attention_embeddings = mh_attention(input_embeddings, input_embeddings, input_embeddings)

    print(f"after_attention_embeddings: ${after_attention_embeddings}")
    print(f"after_attention_embeddings.shape: ${after_attention_embeddings.shape}")
