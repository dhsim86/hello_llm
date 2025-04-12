# 이미지 임베딩 예시

from PIL import Image
from sentence_transformers import SentenceTransformer, util

# OpenAI가 개발한 텍스트-이미지 멀티 모델 모델
# CLIP(Contrastive Language-Image Pre-training) 모델을 사용
# 이미지와 텍스트의 임베딩을 동일한 벡터 공간상에 배치하여
# 유사한 텍스트와 이미지를 찾을 수 있다.
model = SentenceTransformer('clip-ViT-B-32')

# 이미지를 임베딩으로 변환
img_embs = model.encode([Image.open('1.png'), Image.open('2.png'), Image.open('3.png'), Image.open('4.png')])

# 텍스트를 임베딩으로 변환
text_embs = model.encode(['Scottish Fold',
                          'A white cat',
                          'dogs seeing outside',
                          'Electric Car on the Open Road'])

img_cosine_scores = util.pytorch_cos_sim(img_embs, img_embs)
text_cosine_scores = util.pytorch_cos_sim(text_embs, text_embs)
img_text_cosine_scores = util.pytorch_cos_sim(img_embs, text_embs)

print(f"img_cosine_scores: {img_cosine_scores}")
print(f"text_cosine_scores: {text_cosine_scores}")
print(f"img_text_cosine_scores: {img_text_cosine_scores}")
