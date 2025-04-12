import os
import chromadb
from openai import OpenAI

os.environ["OPENAI_API_KEY"] = ""  # OpenAI API 키를 입력

openai_client = OpenAI()
chroma_client = chromadb.Client()


def response_text(openai_resp):
    """
    OpenAI API 응답에서 텍스트 추출
    """
    # 응답에서 텍스트 추출
    text = openai_resp.choices[0].message.content
    return text
