from PIL import Image
import requests

import os
from openai import OpenAI

openai_api_key = ''
os.environ["OPENAI_API_KEY"] = openai_api_key
client = OpenAI()


# dall-e-3 모델을 통해 프롬프트 기반으로 이미지 생성
def generate_image_dalle3(prompt):
    # OpenAI 클라이언트를 통해 DALL-E 3 모델을 사용하여 이미지 생성
    response_oai = client.images.generate(
        model="dall-e-3",
        prompt=str(prompt),
        size="1024x1024",
        quality="standard",
        n=1,
    )

    # 생성된 이미지의 URL을 반환
    result = response_oai.data[0].url
    return result


# 생성한 이미지를 저장하고 불러오기F
def get_generated_image(idx, image_url):
    # 생성된 이미지의 URL에서 이미지를 다운로드하고 저장
    generated_image = requests.get(image_url).content
    image_filename = f'{idx}_gen_img.png'
    with open(image_filename, "wb") as image_file:
        image_file.write(generated_image)
    return Image.open(image_filename)


if __name__ == "__main__":
    gpt_4o_prompt = 'The image depicts a majestic lion sitting in a grassy and flowery landscape. The lion has a unique, artistic mane made of vibrantly colored peacock feathers, giving it a striking and fantastical appearance. The background is bright and serene, with soft blue skies and assorted flowers adding to the tranquil setting. The overall image combines elements of realism and fantasy.'
    original_prompt = 'cute fluffy baby cat rabbit lion hybrid mixed creature character concept, with long flowing mane blowing in the wind, long peacock feather tail, wearing headdress of tribal peacock feathers and flowers, detailed painting, renaissance, 4 k '
    similar_prompt = 'cute fluffy bunny cat lion hybrid mixed creature character concept, with long flowing mane blowing in the wind, long peacock feather tail, wearing headdress of tribal peacock feathers and flowers, detailed painting, renaissance, 4 k'

    # GPT-4o가 만든 프롬프트로 이미지 생성
    gpt_described_image_url = generate_image_dalle3(gpt_4o_prompt)
    gpt4o_prompt_image = get_generated_image(0, gpt_described_image_url)

    # 원본 프롬프트로 이미지 생성
    original_prompt_image_url = generate_image_dalle3(original_prompt)
    original_prompt_image = get_generated_image(1, original_prompt_image_url)

    # 이미지 임베딩으로 검색한 유사 프롬프트로 이미지 생성
    searched_prompt_image_url = generate_image_dalle3(similar_prompt)
    searched_prompt_image = get_generated_image(2, searched_prompt_image_url)
