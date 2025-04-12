from datasets import load_dataset

from llama_index.core import VectorStoreIndex, ServiceContext
from llama_index.core import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# KLUE MRC 데이터셋을 로드
klue_mrc_dataset = load_dataset("klue", "mrc", split="train")
# 1,000개의 샘플을 추출
klue_mrc_dataset = klue_mrc_dataset.train_test_split(train_size=1000, shuffle=False)['train']

# 허깅페이스의 임베딩 모델을 로드
embed_model = HuggingFaceEmbedding(model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS")
service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=None)

# 로컬 모델도 지원
# service_context = ServiceContext.from_defaults(embed_model="local")

text_list = klue_mrc_dataset[:100]['context']
documents = [Document(text=t) for t in text_list]

# Document를 벡터로 변환 후 인덱스를 생성
index_llama = VectorStoreIndex.from_documents(documents, service_context=service_context)

# 기사 본문을 저장한 index를 검색에 사용할 수 있도록 as_retriever 메서드로 검색 엔진으로 변환
retireval_engine = index_llama.as_retriever(similarity_top_k=3, verbose=True)

# 질문을 벡터 검색 엔진에 전달하여 유사 검색
response = retireval_engine.retrieve("이번 연도에는 언제 비가 많이 올까?")

# 검색된 기사의 개수 출력
print(f"document count: {len(response)}")

# 검색된 기사의 내용 출력
for i, res in enumerate(response):
    print(f"document {i}: {res.node.text}")
