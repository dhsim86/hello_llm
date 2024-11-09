import torch
import torch.nn as nn

# 띄어쓰기 단위로 분리
input_text = "나는 최근 파리 여행을 다녀왔다"
input_text_list = input_text.split()
print("input_text_list: ", input_text_list)

# 토큰 -> 아이디 딕셔너리와 아이디 -> 토큰 딕셔너리 만들기
str2idx = {word: idx for idx, word in enumerate(input_text_list)}
idx2str = {idx: word for idx, word in enumerate(input_text_list)}
print("str2idx: ", str2idx)
print("idx2str: ", idx2str)

# 토큰을 토큰 아이디로 변환
input_ids = [str2idx[word] for word in input_text_list]
print("input_ids: ", input_ids)

embedding_dim = 16
embed_layer = nn.Embedding(len(str2idx), embedding_dim) # 사전 크기가 5이고, 16차원의 임베딩을 생성하는 임베딩 레이어 생성

input_embeddings = embed_layer(torch.tensor(input_ids))
input_embeddings = input_embeddings.unsqueeze(0)
print(f"input_embeddings: {input_embeddings}")
print(f"input_embeddings Shape: {input_embeddings.shape}")


