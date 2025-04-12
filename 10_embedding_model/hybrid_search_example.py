# 상호 순위 조합 구현

from collections import defaultdict
from typing import List
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from transformers import AutoTokenizer
from bm25 import BM25


# 각 검색 방식으로 계산해 정해진 문서의 순위를 입력받는다.
# 상호 순위 조합 점수가 높은 순대로 정렬해서 반환
def reciprocal_rank_fusion(rankings: List[List[int]], k=5):
    rrf = defaultdict(float)
    for ranking in rankings:
        for i, doc_id in enumerate(ranking, 1):
            # 각각의 문서 인덱스에 1(k + 순위)의 점수를 더한다.
            rrf[doc_id] += 1.0 / (k + i)
    # 점수를 종합한 딕셔너리(rff)를 점수에 따라 높은 순으로 정렬 후 반환
    return sorted(rrf.items(), key=lambda x: x[1], reverse=True)


rankings = [[1, 4, 3, 5, 6], [2, 1, 3, 6, 4]]
print(reciprocal_rank_fusion(rankings))

# KLUE MRC 데이터셋을 로드
klue_mrc_dataset = load_dataset("klue", "mrc", split="train")
# 1,000개의 샘플을 추출
klue_mrc_dataset = klue_mrc_dataset.train_test_split(train_size=1000, shuffle=False)['train']

# 한국어 문장 임베딩 모델인 snunlp/KR-SBERT-V40K-klueNLI-augSTS를 사용
sentence_model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

# 기사 본문 데이터를 저장하고 있는 context 컬럼을
# 문장 임베딩 모델에 입력하여 문장 임베딩으로 변환
embeddings = sentence_model.encode(klue_mrc_dataset['context'])

import faiss

# 인덱스 생성
# 임베딩 데이터를 저장할 인덱스를 생성
# IndextFlatL2는 KNN(K-Nearest Neighbor) 알고리즘을 사용
index = faiss.IndexFlatL2(embeddings.shape[1])

# 인덱스에 임베딩 저장
index.add(embeddings)

tokenizer = AutoTokenizer.from_pretrained('klue/roberta-base')
bm25 = BM25(klue_mrc_dataset['context'], tokenizer)


# 의미 검색에서 반복적으로 수행하던 검색 쿼리 문장의 임베딩 변환과
# 인덱스 검색을 한 번에 수행
def dense_vector_search(query: str, k: int):
    query_embedding = sentence_model.encode([query])
    distances, indices = index.search(query_embedding, k)
    return distances[0], indices[0]


# 검색 쿼리 문장과 상호 순위 조합에 쓸 파라미터 k를 입력
def hybrid_search(query, k=20):
    _, dense_search_ranking = dense_vector_search(query, 100)  # 의미 검색
    _, bm25_search_ranking = bm25.get_top_k(query, 100)  # 키워드 검색

    # 상호 순위 조합
    results = reciprocal_rank_fusion([dense_search_ranking, bm25_search_ranking], k=k)
    return results


query = "이번 연도에는 언제 비가 많이 올까?"
print("검색 쿼리 문장: ", query)
results = hybrid_search(query)
for idx, score in results[:3]:
    print(klue_mrc_dataset['context'][idx][:50])

print("=" * 80)
query = klue_mrc_dataset[3]['question']  # 로버트 헨리 딕이 1946년에 매사추세츠 연구소에서 개발한 것은 무엇인가?
print("검색 쿼리 문장: ", query)

results = hybrid_search(query)
for idx, score in results[:3]:
    print(klue_mrc_dataset['context'][idx][:50])
