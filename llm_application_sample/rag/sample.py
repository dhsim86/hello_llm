import os
import json

os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"  # OpenAI API 키를 입력


# datasets 라이브러릴 통해 KLUE MRC 데이터셋 다운로드 및 확인
def load_klue_mrc_dataset():
    from datasets import load_dataset

    return load_dataset('klue', 'mrc', split='train')


# 데이터셋으로부터 임베딩 벡터를 만들어 벡터 데이터베이스에 저장
# 라마인덱스는 다음을 디폴트로 사용
# - 임베딩 모델: OpenAI의 text-embedding-ada-002
# - 벡터 데이터베이스: 인메모리 방식의 벡터 데이터베이스
def save_to_vector(dataset):
    from llama_index.core import Document, VectorStoreIndex

    # 데이터셋 중 100개
    text_list = dataset[:100]['context']

    # Document 클래스의 text 인자에 맥락 데이터를 전달
    documents = [Document(text=t) for t in text_list]

    # Document를 벡터로 변환 후 인덱스를 생성
    # 라마인덱스가 내부적으로 텍스트를 임베딩 벡터로 변환 후 인메모리 벡터 데이터베이스에 저장
    # VectorStoreIndex 클래스의 객체를 리턴하는데, 이 객체를 통해 쿼리를 수행
    return VectorStoreIndex.from_documents(documents)


if __name__ == '__main__':
    dataset = load_klue_mrc_dataset()
    print(json.dumps(dataset[0], ensure_ascii=False))

    index = save_to_vector(dataset)

    ##############################################################################################################
    # 벡터 데이터베이스에서 쿼리와 유사한 정보를 벡터 검색
    print(f"question: {dataset[0]['question']}")

    # 기사 본문을 저장한 index를 벡터 검색에 사용할 수 있도록 as_retriever 메서드로 검색 엔진으로 변환
    # similarity_top_k: 유사도 점수가 높은 상위 k개의 결과를 리턴, 가장 가까운 5개의 기사를 반환
    retrieval_engine = index.as_retriever(similarity_top_k=5, verbose=True)

    # 질문을 벡터 검색 엔진에 전달하여 유사한 기사를 검색
    response = retrieval_engine.retrieve(dataset[0]['question'])

    # 검색된 기사의 개수 출력
    print(f"document count: {len(response)}")

    # 검색된 기사의 내용 출력
    for i, res in enumerate(response):
        print(f"document {i}: {res.node.text}")

    ##############################################################################################################
    # 검색한 본문을 활용해 LLM의 답변을 생성

    # index를 쿼리 엔진으로 변환
    query_engine = index.as_query_engine(similarity_top_k=1)
    # query 메서드로 질문을 입력하면 질문과 관련된 기사 본문을 찾아,
    # 프롬프트에 추가 후 LLM에 전달하여 답변받는다.
    response = query_engine.query(dataset[0]['question'])

    print(f"query result: {response}")
