import torch.nn as nn

from transformer_examples.attention.attention_test import compute_attention
from transformer_examples.tokens import embedding_text


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
        print(f"B: {B}, T: {T}, C: {C}")

        # 쿼리와 키, 값을 n_head로 쪼갠다.
        queries = self.weight_q(queries).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        keys = self.weight_k(keys).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        values = self.weight_v(values).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # 각각의 어텐션을 계산
        attention = compute_attention(queries, keys, values, self.is_causal)

        # 입력과 같은 형태로 다시 변환
        output = attention.transpose(1, 2).contiguous().view(B, T, C)

        # 마지막 선형 층을 통과시키고 최종 결과를 반환
        output = self.concat_linear(output)

        return output


if __name__ == '__main__':
    n_head = 4
    input_embeddings = embedding_text.get_input_embeddings()

    mh_attention = MultiHeadAttention(16, 16, n_head)
    after_attention_embeddings = mh_attention(input_embeddings, input_embeddings, input_embeddings)

    print(f"after_attention_embeddings: ${after_attention_embeddings}")
