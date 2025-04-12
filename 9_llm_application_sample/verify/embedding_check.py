import os
from nemoguardrails import LLMRails, RailsConfig
import nest_asyncio

nest_asyncio.apply()

os.environ["OPENAI_API_KEY"] = ""


# 사용자가 인사를 하면 어떻게 반응할지 정의
def greet_user():
    # user greeting: 사용자의 인사 요청을 정의
    # --> NeMo-Guardrails 라이브러리는 세 문장을 임베딩 벡터로 변환 후 저장
    #     앞으로 비슷한 요청이 들어오면 인사로 판단한다.
    # bot express greeting / bot offer help: 요청에 대한 응답 정의
    # flow greeting: 사용자가 인사했을 때, 먼저 인사를 하고 어떤 도움이 필요한지 묻도록 흐름을 정의
    colang_content = """
define user greeting
    "안녕!"
    "How are you?"
    "What's up?"

define bot express greeting
    "안녕하세요!"

define bot offer help
    "어떤걸 도와드릴까요?"

define flow greeting
    user express greeting
    bot express greeting
    bot offer help
"""
    # 언어 모델로 gpt-3.5-turbo를 사용하고, 임베딩 모델은 text-embedding-ada-002로 지정
    yaml_content = """
models:
  - type: main
    engine: openai
    model: gpt-3.5-turbo

  - type: embeddings
    engine: openai
    model: text-embedding-ada-002
"""

    # Rails 설정하기
    # 요청과 응답 흐름 및 모델 정보를 설정
    config = RailsConfig.from_content(
        colang_content=colang_content,
        yaml_content=yaml_content
    )
    # Rails 인스턴스 생성
    rails = LLMRails(config)

    # 사용자가 인사하면 설정한대로, 먼저 인사하고 어떤 도움이 필요한지 묻는다.
    print(rails.generate(messages=[{"role": "user", "content": "안녕하세요!"}]))


# 특정 분야(요리)에 대한 질문이나 요청에 답변하지 않도록 정의
def deny_cooking():
    # user ask about cooking: 사용자가 요리에 대한 질문을 정의
    # bot refuse to respond about cooking: 요리에 대한 질문에 답변하지 않도록 정의
    # flow cooking: 사용자가 요리에 대한 질문을 했을 때, 답변하지 않도록 흐름을 정의
    colang_content_cooking = """
define user ask about cooking
    "How can I cook pasta?"
    "How much do I have to boil pasta?"
    "파스타 만드는 법을 알려줘."
    "요리하는 방법을 알려줘."

define bot refuse to respond about cooking
    "죄송합니다. 저는 요리에 대한 정보는 답변할 수 없습니다. 다른 질문을 해주세요."

define flow cooking
    user ask about cooking
    bot refuse to respond about cooking
"""
    yaml_content = """
models:
  - type: main
    engine: openai
    model: gpt-3.5-turbo

  - type: embeddings
    engine: openai
    model: text-embedding-ada-002
"""

    # initialize rails config
    config = RailsConfig.from_content(
        colang_content=colang_content_cooking,
        yaml_content=yaml_content
    )
    # create rails
    rails_cooking = LLMRails(config)

    print(rails_cooking.generate(messages=[{"role": "user", "content": "사과 파이는 어떻게 만들어?"}]))


if __name__ == '__main__':
    # greet_user()
    deny_cooking()
