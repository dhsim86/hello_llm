import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM


# 모델 ID를 받아 토크나이저 및 모델을 로드 후 파이프라인으로 리턴
def make_inference_pipeline(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", load_in_4bit=True,
                                                 bnb_4bit_compute_dtype=torch.float16)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return pipe


model_id = 'beomi/Yi-Ko-6B'
hf_pipe = make_inference_pipeline(model_id)

if __name__ == '__main__':
    example = """당신은 SQL을 생성하는 SQL 봇입니다. DDL의 테이블을 활용한 Question을 해결할 수 있는 SQL 쿼리를 생성하세요.
    
    ### DDL:
    CREATE TABLE players (
      player_id INT PRIMARY KEY AUTO_INCREMENT,
      username VARCHAR(255) UNIQUE NOT NULL,
      email VARCHAR(255) UNIQUE NOT NULL,
      password_hash VARCHAR(255) NOT NULL,
      date_joined DATETIME NOT NULL,
      last_login DATETIME
    );
    
    ### Question:
    사용자 이름에 'admin'이 포함되어 있는 계정의 수를 알려주세요.
    
    ### SQL:
    """

    # 프롬프트를 넣어 결과를 확인
    result = hf_pipe(example, do_sample=False,
                     return_full_text=False, max_length=512, truncation=True)
    print(result[0]['generated_text'])
