from llama_index.core import get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

from sample import load_klue_mrc_dataset, save_to_vector

if __name__ == '__main__':
    dataset = load_klue_mrc_dataset()
    index = save_to_vector(dataset)

    # 검색을 위한 Retriever 생성
    # VectorIndexRetriever 클래스를 사용해 벡터 데이터베이스에서 검색하는 retriever 생성
    retriever = VectorIndexRetriever(index=index,
                                     similarity_top_k=1)

    # 검색 결과를 질문과 결합하는 synthesizer, 프롬프트를 통합할 때 사용
    response_synthesizer = get_response_synthesizer()

    # RetrieverQueryEngine 클래스를 통해 검색 증강 생성을 한 번에 수행하는
    # query_engine 생성
    # SimilarityPostprocessor를 통해 질문과 유사도가 낮을 경우 필터링하도록 설정
    # --> 예제에서는 유사도가 0.7이하인 문서는 필터링하도록 설정
    query_engine = RetrieverQueryEngine(retriever=retriever,
                                        response_synthesizer=response_synthesizer,
                                        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)])

    # RAG 수행
    response = query_engine.query('북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은?')
    print(response)
