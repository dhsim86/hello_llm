# CLIP 모델은 허깅페이스 트랜스포머 라이브러리를 통해 쉽게 활용 가능

from transformers import CLIPProcessor, CLIPModel

# 모델과 데이터셋 처리 프로세서를 로드
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# 이미지를 URL을 통해 다운로드하고, 모델에 입력
import requests
from PIL import Image

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# 제로샷 추론
# 입력 텍스트는 'a photo of {사물}' 형태로 넣어준다.
inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)
print(probs)
