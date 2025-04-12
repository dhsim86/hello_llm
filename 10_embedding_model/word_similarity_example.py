# sentence_transformers 및 사이킷런의 cosine_similarity를 통해 코사인 유사도를 테스트
# 각 단어의 유사성을 통해 단어나 문장 사이의 관계를 계산할 수 있다.
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

smodel = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
dense_embeddings = smodel.encode(['학교', '공부', '운동'])
print(cosine_similarity(dense_embeddings))