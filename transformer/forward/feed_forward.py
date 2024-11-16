from transformer.tokens import embedding_text

import torch.nn as nn


class PreLayerNormFeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout):
        super().__init__()

        # 선형층 1,2
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # 드롭아웃층 1,2
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # 활성 함수
        self.activation = nn.GELU()

        # 층 정규화
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src):
        x = self.norm(src)
        x = x + self.linear2(self.dropout1(self.activation(self.linear1(x))))
        x = self.dropout2(x)

        return x


if __name__ == '__main__':
    input_embeddings = embedding_text.get_input_embeddings()
    pre_layer_norm_feed_forward = PreLayerNormFeedForward(16, 32, 0.2)

    result = pre_layer_norm_feed_forward(input_embeddings)
    print(f"result: {result}")
