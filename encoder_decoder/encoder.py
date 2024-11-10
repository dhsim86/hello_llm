import copy

from attention.multi_head_attention import MultiHeadAttention
from forward.feed_forward import PreLayerNormFeedForward
from tokens import embedding_text

import torch.nn as nn


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, dim_feedforward, dropout):
        super().__init__()

        # 멀티 헤드 어텐션
        self.attn = MultiHeadAttention(d_model, d_model, n_head)

        # 층 정규화
        self.norm1 = nn.LayerNorm(d_model)

        # 드롭아웃
        self.dropout1 = nn.Dropout(dropout)

        # 피드 포워드
        self.feed_forward = PreLayerNormFeedForward(d_model, dim_feedforward, dropout)

    def forward(self, src):
        # 층 정규화 (사전 정규화)
        norm_x = self.norm1(src)

        # 멀티 헤드 어텐션
        attn_output = self.attn(norm_x, norm_x, norm_x)

        # 잔차 연결 (드롭아웃한 어텐션 결과 + 입력)
        x = src + self.dropout1(attn_output)

        # 피드 포워드 연산
        x = self.feed_forward(x)

        return x


def get_clone(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        # 인코더 중첩
        self.layers = get_clone(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src):
        output = src

        # 인코더 결과를 다시 다음 인코더의 입력으로
        for mod in self.layers:
            output = mod(output)

        return output


if __name__ == '__main__':
    input_embeddings = embedding_text.get_input_embeddings()

    encoder_layer = TransformerEncoderLayer(16, 4, 32, 0.2)
    # result = encoder_layer(input_embeddings)

    encoder = TransformerEncoder(encoder_layer, 4)
    result = encoder(input_embeddings)
    print(f"result: {result}")
