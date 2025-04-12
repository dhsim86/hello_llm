from client import openai_client, response_text
import time


class OpenAICache:
    def __init__(self, openai_client):
        self.openai_client = openai_client

        # 일치 캐시로 딕셔너리 사용
        self.cache = {}

    def generate(self, prompt):
        # 딕셔너리(캐시)에 없다면 새롭게 텍스트를 생성
        # 있다면 바로 리턴
        if prompt not in self.cache:
            responses = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            self.cache[prompt] = response_text(responses)
        return self.cache[prompt]


if __name__ == '__main__':
    openai_cache = OpenAICache(openai_client)

    question = "북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은?"

    for _ in range(2):
        start_time = time.time()
        response = openai_cache.generate(question)
        print(f"질문: {question}")
        print(f"소요 시간: {'{:.2f}s'.format(time.time() - start_time)}")
        print(f"답변: {response}")
        print("")
