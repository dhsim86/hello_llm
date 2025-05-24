from datasets import load_dataset

import os
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

pinecone_api_key = ""
openai_api_key = ''

pc = Pinecone(api_key=pinecone_api_key)
os.environ["OPENAI_API_KEY"] = openai_api_key
client = OpenAI()


def create_pinecone_index(index_name):
    # 인덱스 생성
    index_name = "llm-multimodal"
    try:
        pc.create_index(
            name=index_name,
            dimension=512,  # 임베딩 벡터의 차원 수
            metric="cosine",  # 코사인 유사도를 평가 지표로 사용
            spec=ServerlessSpec(
                "aws", "us-east-1"
            )
        )
        print(pc.list_indexes())
    except:
        print("Index already exists")
    index = pc.Index(index_name)  # 인덱스

    return index


# 데이터셋에 있는 이미지 생성용 프롬프트들을 임베딩 벡터로 변환
def convert_prompt_to_embedding(dataset):
    import torch
    from tqdm.auto import trange
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, CLIPTextModelWithProjection

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # openai/clip-vit-base-patch32 모델을 사용하여 텍스트 임베딩을 생성
    # 텍스트 모델과 토크나이저를 로드
    text_model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    # 토큰화
    tokens = tokenizer(dataset['prompt'], padding=True, return_tensors="pt", truncation=True)

    batch_size = 16
    text_embs = []

    # 프롬프트들을 텍스트 임베딩으로 변환
    for start_idx in trange(0, len(dataset), batch_size):
        with torch.no_grad():
            outputs = text_model(input_ids=tokens['input_ids'][start_idx:start_idx + batch_size],
                                 attention_mask=tokens['attention_mask'][start_idx:start_idx + batch_size])
            text_emb_tmp = outputs.text_embeds
        text_embs.append(text_emb_tmp)
    text_embs = torch.cat(text_embs, dim=0)

    # 1,000개의 512차원 임베딩 벡터 생성
    text_embs.shape  # (1000, 512)

    return text_embs


# Pinecone에 임베딩 벡터 저장
def save_embeddings_to_pinecone(index, text_embs, dataset):
    input_data = []
    for id_int, emb, prompt in zip(range(0, len(dataset)), text_embs.tolist(), dataset['prompt']):
        input_data.append(
            {
                "id": str(id_int),
                "values": emb,
                "metadata": {
                    "prompt": prompt
                }
            }
        )

    index.upsert(
        vectors=input_data
    )


if __name__ == "__main__":
    # 인덱스 생성
    # index = create_pinecone_index("llm-multimodal")
    print(pc.list_indexes())

    index = pc.Index("llm-multimodal")  # 인덱스

    dataset = load_dataset("poloclub/diffusiondb", "2m_first_1k", split='train')
    text_embs = convert_prompt_to_embedding(dataset)

    save_embeddings_to_pinecone(index, text_embs, dataset)
