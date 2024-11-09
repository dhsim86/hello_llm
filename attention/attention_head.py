from attention.attention_test import compute_attention
from tokens import embedding_text

import torch.nn as nn


class AttentionHead(nn.Module):
    def __init__(self, token_embed_dim, head_dim, is_causal=False):
        super().__init__()
        self.is_causal = is_causal
        self.weight_q = nn.Linear(token_embed_dim, head_dim)    # 쿼리 벡터 생성을 위한 선형 층
        self.weight_k = nn.Linear(token_embed_dim, head_dim)    # 키 벡터 생성을 위한 선형 층
        self.weight_v = nn.Linear(token_embed_dim, head_dim)    # 값 벡터 생성을 위한 선형 층

    def forward(self, queries, keys, values):
        outputs = compute_attention(self.weight_q(queries),     # 쿼리 벡터 생성
                                    self.weight_k(keys),        # 키 벡터 생성
                                    self.weight_v(values),      # 값 벡터 생성
                                    is_causal=self.is_causal)
        return outputs


if __name__ == '__main__':
    input_embeddings = embedding_text.get_input_embeddings()

    attention_head = AttentionHead(16, 16)
    after_attention_embeddings = attention_head(input_embeddings,
                                                input_embeddings,
                                                input_embeddings)

    print(f"after_attention_embeddings: ${after_attention_embeddings}")
