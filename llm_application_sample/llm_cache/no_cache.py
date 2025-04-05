from client import openai_client, response_text
import time

if __name__ == '__main__':
    question = "북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은?"

    # LLM 캐시를 사용하지 않았을 경우의 예제
    # 각 LLM 출력마다 생성 시간이 소요된다.
    for _ in range(2):
        start_time = time.time()
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": question
                }
            ]
        )
        response = response_text(response)
        print(f"질문: {question}")
        print(f"소요 시간: {'{:.2f}s'.format(time.time() - start_time)}")
        print(f"답변: {response}")
        print("")
