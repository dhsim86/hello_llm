import copy

from attention.attention_test import compute_attention, get_qkv
from attention.multi_head_attention import MultiHeadAttention
from forward.feed_forward import PreLayerNormFeedForward
from tokens import embedding_text

import torch.nn as nn


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, dim_feedforward=2048, dropout=0.1):
        super().__init__()

        # 디코더 셀프 어텐션
        self.self_attn = MultiHeadAttention(d_model, d_model, n_head)

        # 멀티 헤드 어텐션 (인코더 결과 + 디코더 입력)
        self.multi_head_attn = MultiHeadAttention(d_model, d_model, n_head)

        self.feed_forward = PreLayerNormFeedForward(d_model, dim_feedforward, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, tgt, encoder_output, is_causal=True):
        # 디코더 입력 -> 층 정규화
        x = self.norm1(tgt)

        # 디코더 입력 -> 어텐션
        # 잔차 연결
        x = x + self.dropout1(self.self_attn(x, x, x))

        # 크로스 어텐션
        ## 다시 층 정규화
        x = self.norm2(x)
        ## 인코더 결과를 입력으로 멀티 헤드 어텐션
        ## 디코더 결과와 잔차 연결
        x = x + self.dropout2(self.multi_head_attn(x, encoder_output, encoder_output, is_causal=is_causal))

        # 피드 포워드
        x = self.feed_forward(x)

        return x


def get_clone(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        # 디코더 중첩
        self.layers = get_clone(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt, src):
        output = tgt

        for mod in self.layers:
            output = mod(tgt, src)

        return output


if __name__ == '__main__':
    input_embeddings = embedding_text.get_input_embeddings()

    queries, keys, values = get_qkv(embedding_text.get_input_embeddings())
    encoder_result = compute_attention(queries, keys, values)

    decoder_layer = TransformerDecoderLayer(16, 4)
    result = decoder_layer(input_embeddings, encoder_result)

    print(f"result: {result}")
