def create_pinecone_index():
    from pinecone import Pinecone, ServerlessSpec

    pinecone_api_key = ""
    pc = Pinecone(api_key=pinecone_api_key)

    # 인덱스의 이름과 임베딩 벡터의 차원수를 지정하여 인덱스를 생성
    pc.create_index("llm-book", spec=ServerlessSpec("aws", "us-east-1"), dimension=768)

    # 인덱스를 불러온다
    index = pc.Index('llm-book')


def load_pinecone_index():
    from pinecone import Pinecone, ServerlessSpec

    pinecone_api_key = ""
    pc = Pinecone(api_key=pinecone_api_key)

    return pc.Index('llm-book')


def create_embedding():
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    # 임베딩 모델 불러오기
    sentence_model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

    # 데이터셋 불러오기
    klue_dp_train = load_dataset('klue', 'dp', split='train[:100]')

    # sentence 컬럼을 임베딩 벡터로 변환
    embeddings = sentence_model.encode(klue_dp_train['sentence'])

    return klue_dp_train, embeddings


# 파인콘에서 쓰는 데이터 형식으로 변환
def prepare_to_pipecone_data(dataset, embeddings):
    # 넘파이 라이브러리의 데이터 타입인 embeddings를 파이썬 기본 데이터 타입으로 변경
    embeddings = embeddings.tolist()

    # 파인콘 인덱스에 저장하기 위한 데이터 형식으로 변환
    # id, values, metadata로 구성된 딕셔너리 형태로 데이터를 변환해야 한다.
    # {"id": 문서 ID(str), "values": 벡터 임베딩(List[float]), "metadata": 메타 데이터(dict)} 형태로 데이터 준비
    insert_data = []
    for idx, (embedding, text) in enumerate(zip(embeddings, dataset['sentence'])):
        insert_data.append({"id": str(idx), "values": embedding, "metadata": {'text': text}})

    return insert_data


if __name__ == "__main__":
    # create_pinecone_index()

    index = load_pinecone_index()
    klue_dp_train, embeddings = create_embedding()
    insert_data = prepare_to_pipecone_data(klue_dp_train, embeddings)

    # 인덱스에 임베딩 벡터 저장
    # 파인콘에서는 하나의 인덱스안에서도 여러 네임스페이스를 구분할 수 있다.
    # index.upsert(vectors=insert_data, namespace='llm-book-sub')
    print(f"query: {embeddings[0].tolist()}")
    query_response = index.query(
        namespace='llm-book-sub',  # 검색할 네임스페이스
        top_k=10,  # 몇 개의 결과를 반환할지
        include_values=True,  # 벡터 임베딩 반환 여부
        include_metadata=True,  # 메타 데이터 반환 여부
        vector=embeddings[0].tolist()  # 검색할 벡터 임베딩
    )
    print(query_response)

    # 문서의 ID를 기반으로 데이터를 수정하고 삭제
    from sentence_transformers import SentenceTransformer

    # 임베딩 모델 불러오기
    sentence_model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
    new_text = "이것은 새로운 문서입니다."

    new_embedding = sentence_model.encode([new_text])[0].tolist()
    update_response = index.update(id='0',
                                   values=new_embedding,
                                   metadata={'text': new_text},
                                   namespace='llm-book-sub')

    delete_response = index.delete(ids=['0'], namespace='llm-book-sub')
