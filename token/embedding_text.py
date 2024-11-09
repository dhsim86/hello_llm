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
token_ids = [str2idx[word] for word in input_text_list]
print("token_ids: ", token_ids)

embedding_dim = 16
embed_layer = nn.Embedding(len(str2idx), embedding_dim) # 사전 크기가 5이고, 16차원의 임베딩을 생성하는 임베딩 레이어 생성

token_embeddings = embed_layer(torch.tensor(token_ids))
token_embeddings = token_embeddings.unsqueeze(0)

max_positions = 12 # 최대 토큰 수
position_embed_layer = nn.Embedding(max_positions, embedding_dim) # 위치 임베딩 레이어 생성
position_ids = torch.arange(len(token_ids), dtype=torch.long).unsqueeze(0)

position_encodings = position_embed_layer(position_ids)

input_embeddings = token_embeddings + position_encodings

print(f"token_embeddings: {token_embeddings}")
print(f"position_encodings: {position_encodings}")
print(f"input_embeddings: {input_embeddings}")


