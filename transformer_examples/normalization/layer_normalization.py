from transformer_examples.tokens import embedding_text

import torch.nn as nn

if __name__ == '__main__':
    embedding_dim = 16
    norm = nn.LayerNorm(embedding_dim)

    input_embeddings = embedding_text.get_input_embeddings()

    norm_x = norm(input_embeddings)

    # 정규화전 평균 및 표준편차
    print(f"PRE Normalization -  mean: {input_embeddings.mean(dim=-1).data}, std: {input_embeddings.std(dim=-1).data}")

    # 정규화후 평균 및 표준편차
    print(f"AFTER Normalization -  mean: {norm_x.mean(dim=-1).data}, std: {norm_x.std(dim=-1).data}")
