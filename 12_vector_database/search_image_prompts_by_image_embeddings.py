from datasets import load_dataset
from transformers import AutoProcessor, CLIPVisionModelWithProjection

from pinecone import Pinecone

pinecone_api_key = ""
pc = Pinecone(api_key=pinecone_api_key)


# 데이터셋 다운로드
def get_data_from_dataset():
    dataset = load_dataset("poloclub/diffusiondb", "2m_first_1k", split='train')

    example_index = 867

    # 이미지 컬럼과 이미지를 만들 때 사용한 프롬프트 컬럼을 사용
    original_image = dataset[example_index]['image']
    original_prompt = dataset[example_index]['prompt']

    return original_image, original_prompt


if __name__ == "__main__":
    original_image, original_prompt = get_data_from_dataset()

    # 이미지 임베딩 모델을 로드
    vision_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

    inputs = processor(images=original_image, return_tensors="pt")

    # 이미지를 이미지 임베딩으로 변환
    outputs = vision_model(**inputs)
    image_embeds = outputs.image_embeds

    # 파인콘 벡터 데이터베이스에서 이미지 임베딩으로 프롬프트 검색
    index = pc.Index("llm-multimodal")
    search_results = index.query(
        vector=image_embeds[0].tolist(),
        top_k=3,
        include_values=False,
        include_metadata=True
    )
    searched_idx = int(search_results['matches'][0]['id'])

    print(f"검색된 프롬프트 3개: {search_results}")
    print(f"검색된 프롬프트의 인덱스: {searched_idx}")
