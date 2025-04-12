from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# KLUE MRC 데이터셋을 로드
klue_mrc_dataset = load_dataset("klue", "mrc", split="train")

# 한국어 문장 임베딩 모델인 snunlp/KR-SBERT-V40K-klueNLI-augSTS를 사용
sentence_model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

# 1,000개의 샘플을 추출
klue_mrc_dataset = klue_mrc_dataset.train_test_split(train_size=1000, shuffle=False)['train']

# 기사 본문 데이터를 저장하고 있는 context 컬럼을
# 문장 임베딩 모델에 입력하여 문장 임베딩으로 변환
embeddings = sentence_model.encode(klue_mrc_dataset['context'])

print(f"embedding.shape: {embeddings.shape}")

import faiss

# 인덱스 생성
# 임베딩 데이터를 저장할 인덱스를 생성
# IndextFlatL2는 KNN(K-Nearest Neighbor) 알고리즘을 사용
index = faiss.IndexFlatL2(embeddings.shape[1])

# 인덱스에 임베딩 저장
index.add(embeddings)

# 의미 검색
query = "이번 연도에는 언제 비가 많이 올까?"
query_embedding = sentence_model.encode([query])

# 쿼리 임베딩과 가장 가까운 3개의 문서를 검색
distances, indices = index.search(query_embedding, k=3)

print(f"distances: {distances}")
for idx in indices[0]:
    print(f" context: {klue_mrc_dataset['context'][idx][:50]}")

# 의미 검색의 한계

# 로버트 헨리 딕이 1946년에 매사추세츠 연구소에서 개발한 것은 무엇인가?
query = klue_mrc_dataset[3]['question']
query_embedding = sentence_model.encode([query])
distances, indices = index.search(query_embedding, k=3)

print(f"distances: {distances}")
for idx in indices[0]:
    print(f" context: {klue_mrc_dataset['context'][idx][:50]}")
