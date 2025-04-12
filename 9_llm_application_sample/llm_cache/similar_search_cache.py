import os

from client import openai_client, chroma_client, response_text
import time

from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# 임베딩 모델로 OpenAI의 text-embedding-ada-002 사용
openai_ef = OpenAIEmbeddingFunction(api_key=os.environ["OPENAI_API_KEY"],
                                    model_name="text-embedding-ada-002")

# 유사 검색 캐시를 chroma_client로 생성
# 컬렉션(테이블) 생성시 embedding_function 인자로 임베딩 모델을 등록
# 텍스트를 전달하면 등록된 임베딩 모델을 통해 벡터로 변환
semantic_cache = chroma_client.create_collection(name="semantic_cache",
                                                 embedding_function=openai_ef,
                                                 metadata={"hnsw:space": "cosine"})


class OpenAICache:
    def __init__(self, openai_client, semantic_cache):
        self.openai_client = openai_client

        self.semantic_cache = semantic_cache

    def generate(self, prompt):
        # 크로마 벡터 데이터베이스의 query 메서드에 query_texts 인자로 전달하면
        # 벡터 데이터베이스에 등록된 임베딩 모델을 통해 텍슽를 임베딩 벡터로 변환 후 검색
        similar_doc = self.semantic_cache.query(query_texts=[prompt], n_results=1)

        # 검색 결과가 있고, 검색한 문서와 검색 결과 문서 사이의 거리(distance)가
        # 0.1보다 작으면 검색된 문서를 반환
        similar_result = similar_doc['distances'][0][0] if len(similar_doc['distances'][0]) > 0 else None
        if len(similar_doc['distances'][0]) > 0 and similar_doc['distances'][0][0] < 0.1:
            print("cache hit, distance: ", similar_result)
            return similar_doc['metadatas'][0][0]['response']
        else:
            # 검색 결과가 없거나, 검색된 문서와 검색 결과 문서 사이의 거리(distance)가
            # 0.1보다 크면 OpenAI API를 통해 새롭게 텍스트를 생성
            if similar_result is not None:
                print("cache miss, distance: ", similar_result)
            else:
                print("cache miss")
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            # 생성된 텍스트를 추후에 캐시로 활용할 수 있도록 크로마 벡터 데이터베이스에 저장
            self.semantic_cache.add(documents=[prompt],
                                    metadatas=[{"response": response_text(response)}],
                                    ids=[prompt])
            return response_text(response)


if __name__ == '__main__':
    openai_cache = OpenAICache(openai_client, semantic_cache)

    questions = [
        "북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은?",
        "북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은?",
        "북태평양 기단과 오호츠크해 기단이 만나 한반도에 머무르는 기간은?",
        "북태평양 기단과 오호츠크해 기단 말고 또 다른 기단이 있나요? 성질은?",
        "국내에 북태평양 기단과 오호츠크해 기단이 함께 머무르는 기간은?"]

    for question in questions:
        start_time = time.time()
        response = openai_cache.generate(question)
        print(f"질문: {question}")
        print(f"소요 시간: {'{:.2f}s'.format(time.time() - start_time)}")
        print(f"답변: {response}")
        print("")
