import copy

import torch.nn as nn

from transformer_examples.attention.attention_test import compute_attention, get_qkv
from transformer_examples.attention.multi_head_attention import MultiHeadAttention
from transformer_examples.forward.feed_forward import PreLayerNormFeedForward
from transformer_examples.tokens import embedding_text


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, dim_feedforward=2048, dropout=0.1):
        super().__init__()

        # 디코더 셀프 어텐션 (마스크 멀티 헤드 어텐션)
        # TODO: is_causal 관련 에러 수정
        self.self_attn = MultiHeadAttention(d_model, d_model, n_head, is_causal=True)

        # 멀티 헤드 어텐션 (인코더 결과 + 디코더 입력)
        self.multi_head_attn = MultiHeadAttention(d_model, d_model, n_head)

        self.feed_forward = PreLayerNormFeedForward(d_model, dim_feedforward, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, tgt, encoder_output):
        ###################################
        # 디코더 입력 -> 층 정규화
        x = self.norm1(tgt)

        ###################################
        # 디코더 입력에 대한 마스크 어텐션
        # 1. 디코더 입력 -> 마스크 멀티 헤드 어텐션
        # 2. 잔차 연결
        x = x + self.dropout1(self.self_attn(x, x, x))

        ###################################
        # 크로스 어텐션
        # 1. 다시 층 정규화
        x = self.norm2(x)
        # 2. 인코더 결과를 입력으로 멀티 헤드 어텐션
        #    - 쿼리: 디코더의 잠재 상태 / 키와 값: 인코더의 결과
        # 3. 디코더 결과와 잔차 연결
        x = x + self.dropout2(self.multi_head_attn(x, encoder_output, encoder_output))

        ###################################
        # 피드 포워드
        x = self.feed_forward(x)

        return x


def get_clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        # 디코더 중첩
        self.layers = get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt, src):
        output = tgt
        for idx, mod in enumerate(self.layers):
            print(f"idx: {idx + 1}번째 디코더 블록")
            output = mod(output, src)
        return output


if __name__ == '__main__':
    input_embeddings = embedding_text.get_input_embeddings()

    queries, keys, values = get_qkv(embedding_text.get_input_embeddings())
    encoder_result = compute_attention(queries, keys, values)

    decoder_layer = TransformerDecoderLayer(16, 4)
    decoder = TransformerDecoder(decoder_layer, 4)
    result = decoder(input_embeddings, encoder_result)
    print(f"result: {result}")
