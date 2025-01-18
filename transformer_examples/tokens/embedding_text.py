import torch
import torch.nn as nn


def get_input_embeddings() -> str:
    # 띄어쓰기 단위로 분리
    input_text = "나는 최근 파리 여행을 다녀왔다"
    input_text_list = input_text.split()
    print("input_text_list: ", input_text_list)

    # 토큰 -> 아이디 딕셔너리와 아이디 -> 토큰 딕셔너리(사전) 만들기
    str2idx = {word: idx for idx, word in enumerate(input_text_list)}
    idx2str = {idx: word for idx, word in enumerate(input_text_list)}
    print("str2idx: ", str2idx)
    print("idx2str: ", idx2str)

    # 토큰을 토큰 아이디로 변환
    token_ids = [str2idx[word] for word in input_text_list]
    print("token_ids: ", token_ids)

    ##################################################################
    # 토큰 임베딩 생성
    embedding_dim = 16
    embed_layer = nn.Embedding(len(str2idx), embedding_dim)  # 사전 크기가 5이고, 16차원의 임베딩을 생성하는 임베딩 레이어 생성
    print(f"torch.tensor(token_ids): {torch.tensor(token_ids)}")    # 1차원 텐서 객체 생성
    token_embeddings = embed_layer(torch.tensor(token_ids))
    # 딥러닝 모델의 입력 데이터 형식을 맞추기 위해 배치 차원을 추가
    # 배치 크기를 1로 설정하여 한 번에 하나의 데이터만 처리하겠다는 의미
    token_embeddings = token_embeddings.unsqueeze(0)         # 임의 차원 생성 ([5, 16] => [1, 5, 16])

    ##################################################################
    # 위치 인코딩 생성
    max_positions = 12  # 최대 토큰 수
    position_embed_layer = nn.Embedding(max_positions, embedding_dim)  # 위치 임베딩 레이어 생성
    # 주어진 범위내의 정수 또는 부동 소수점 수의 시퀀스를 생성하여 1차원 텐서 객체 생성
    position_ids = torch.arange(len(token_ids), dtype=torch.long).unsqueeze(0)
    print(f"position_ids: {position_ids}")
    position_encodings = position_embed_layer(position_ids)

    ##################################################################

    input_embeddings = token_embeddings + position_encodings

    print(f"token_embeddings: {token_embeddings}")
    print(f"position_encodings: {position_encodings}")
    print(f"input_embeddings: {input_embeddings}")

    return input_embeddings


if __name__ == '__main__':
    get_input_embeddings()
