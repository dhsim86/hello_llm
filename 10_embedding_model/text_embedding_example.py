# 텍스트 임베딩 예시

from sentence_transformers import SentenceTransformer, util

# 한국어 문장 임베딩 모델인 snunlp/KR-SBERT-V40K-klueNLI-augSTS를 사용
model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

# encode 메서드를 통해 입력 문장을 문장 임베딩으로 변환
embs = model.encode(['잠이 안 옵니다', '졸음이 옵니다.', '기차가 옵니다'])

# 코사인 유사도 계산
cosine_scores = util.pytorch_cos_sim(embs, embs)
print(cosine_scores)
