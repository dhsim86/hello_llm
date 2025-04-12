import math
import numpy as np

from typing import List
from transformers import PreTrainedTokenizer
from collections import defaultdict


class BM25:
    def __init__(self, corpus: List[List[str]], tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.corpus = corpus

        # 토큰화된 텍스트 데이터
        self.tokenized_corpus = self.tokenizer(corpus, add_special_tokens=False)['input_ids']
        self.n_docs = len(self.tokenized_corpus)
        self.avg_doc_lens = sum(len(lst) for lst in self.tokenized_corpus) / len(self.tokenized_corpus)
        self.idf = self._calculate_idf()
        self.term_freqs = self._calculate_term_freqs()

    # 각 토큰이 몇 개의 문서에 등장하는지 집계
    def _calculate_idf(self):
        idf = defaultdict(float)
        # 토큰화된 텍스트 데이터를 순회하면서 토큰 아이디마다
        # 총 몇 개의 문서에 등장하는지 idf 딕셔너리에 저장
        for doc in self.tokenized_corpus:
            for token_id in set(doc):
                idf[token_id] += 1
        for token_id, doc_frequency in idf.items():
            idf[token_id] = math.log(((self.n_docs - doc_frequency + 0.5) / (doc_frequency + 0.5)) + 1)
        return idf

    # 각 토큰이 각 문서 내에서 몇 번 등장하는지 계산
    def _calculate_term_freqs(self):
        # 문서의 수만큼 딕셔너리를 만들고
        # 각 문서 내에 어떤 토큰이 몇 번 등장하는지 집계
        term_freqs = [defaultdict(int) for _ in range(self.n_docs)]
        for i, doc in enumerate(self.tokenized_corpus):
            for token_id in doc:
                term_freqs[i][token_id] += 1
        return term_freqs

    # 저장한 문서와 점수를 계산
    def get_scores(self, query: str, k1: float = 1.2, b: float = 0.75):
        # 검색하려는 쿼리와 각 문서 사이의 점수를 계산
        query = self.tokenizer([query], add_special_tokens=False)['input_ids'][0]
        scores = np.zeros(self.n_docs)
        for q in query:
            idf = self.idf[q]
            for i, term_freq in enumerate(self.term_freqs):
                q_frequency = term_freq[q]
                doc_len = len(self.tokenized_corpus[i])
                score_q = idf * (q_frequency * (k1 + 1)) / (
                        (q_frequency) + k1 * (1 - b + b * (doc_len / self.avg_doc_lens)))
                scores[i] += score_q
        return scores

    # 상위 k개의 점수와 인덱스를 추출
    def get_top_k(self, query: str, k: int):
        # 쿼리와 문서 사이의 점수가 가장 높은 k개의 문서의 인덱스와 점수를 반환
        scores = self.get_scores(query)
        top_k_indices = np.argsort(scores)[-k:][::-1]
        top_k_scores = scores[top_k_indices]
        return top_k_scores, top_k_indices


if __name__ == '__main__':
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('klue/roberta-base')

    bm25 = BM25(['안녕하세요', '반갑습니다', '안녕 서울'], tokenizer)
    print(f"scores: {bm25.get_scores('안녕')}")

    # 키워드 검색의 한계
    from datasets import load_dataset

    # KLUE MRC 데이터셋을 로드
    klue_mrc_dataset = load_dataset("klue", "mrc", split="train")

    # 1,000개의 샘플을 추출
    klue_mrc_dataset = klue_mrc_dataset.train_test_split(train_size=1000, shuffle=False)['train']

    bm25 = BM25(klue_mrc_dataset['context'], tokenizer)
    query = "이번 연도에는 언제 비가 많이 올까?"
    _, bm25_search_ranking = bm25.get_top_k(query, 100)

    for idx in bm25_search_ranking[:3]:
        print(f" context: {klue_mrc_dataset['context'][idx][:50]}")

    # 키워드 검색의 장점
    query = "로버트 헨리 딕이 1946년에 매사추세츠 연구소에서 개발한 것은 무엇인가?"
    _, bm25_search_ranking = bm25.get_top_k(query, 100)

    for idx in bm25_search_ranking[:3]:
        print(f" context: {klue_mrc_dataset['context'][idx][:50]}")
