from datasets import load_dataset

import requests
import base64
from io import BytesIO

import os
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

pinecone_api_key = ""
openai_api_key = ''

pc = Pinecone(api_key=pinecone_api_key)
os.environ["OPENAI_API_KEY"] = openai_api_key
client = OpenAI()


# 데이터셋 다운로드
def get_data_from_dataset():
    dataset = load_dataset("poloclub/diffusiondb", "2m_first_1k", split='train')

    example_index = 867

    # 이미지 컬럼과 이미지를 만들 때 사용한 프롬프트 컬럼을 사용
    original_image = dataset[example_index]['image']
    original_prompt = dataset[example_index]['prompt']

    return original_image, original_prompt


# 이미지를 base64로 인코딩
def make_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str


# GPT-4o 모델로 이미지와 명령 프롬프트를 전달
def generate_description_from_image_gpt4(prompt, image64):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {client.api_key}"
    }
    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }
    response_oai = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    result = response_oai.json()['choices'][0]['message']['content']
    return result


if __name__ == "__main__":
    original_image, original_prompt = get_data_from_dataset()
    print(f"Original Prompt: {original_prompt}")

    # 이미지를 base64로 인코딩
    image_base64 = make_base64(original_image)
    described_result = generate_description_from_image_gpt4("Describe provided image", image_base64)

    print(f"Described Result: {described_result}")
