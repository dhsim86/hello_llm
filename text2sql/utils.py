# DDL 및 사용자 요청사항(question), 정답 SQL(query)을 통해 프롬프트 생성
# 학습할 때는 정답 SQL을 넣고 학습
# 추론할 때는 SQL을 넣지 않는다.
def make_prompt(ddl, question, query=''):
    prompt = f"""당신은 SQL을 생성하는 SQL 봇입니다. DDL의 테이블을 활용한 Question을 해결할 수 있는 SQL 쿼리를 생성하세요.

### DDL:
{ddl}

### Question:
{question}

### SQL:
{query}"""
    return prompt


# 입력한 데이터프레임을 순회하면서 평가를 위해 사용할 프롬프트 생성
# 프롬프트를 jsonl 파일에 기록
# 프롬프트의 내용
#   - DDL과 Question을 바탕으로 LLM이 생성한 SQL과 정답 SQL이 동일한 기능을 하는지 평가
#   - 판단 결과는 JSON 형식으로 resolve_yn 이라는 프로퍼티에 yes / no 값으로 받는다.
def make_requests_for_gpt_evaluation(df, filename, dir='requests'):
    from pathlib import Path
    import json

    if not Path(dir).exists():
        Path(dir).mkdir(parents=True)

    prompts = []
    for idx, row in df.iterrows():
        # 프롬프트 생성
        prompts.append(
            """Based on below DDL and Question, evaluate gen_sql can resolve Question. If gen_sql and gt_sql do equal job, return "yes" else return "no". Output JSON Format: {"resolve_yn": ""}""" + f"""

DDL: {row['context']}
Question: {row['question']}
gt_sql: {row['answer']}
gen_sql: {row['gen_sql']}"""
        )

    # 모델 이름 및 출력 타입을 지정하고, 프롬프트를 system role로 지정
    jobs = [{"model": "gpt-4o", "response_format": {"type": "json_object"},
             "messages": [{"role": "system", "content": prompt}]} for prompt in prompts]

    # jsonl 형식으로 요청 정보를 저장
    with open(Path(dir, filename), "w") as f:
        for job in jobs:
            json_string = json.dumps(job)
            f.write(json_string + "\n")


if __name__ == '__main__':
    from datasets import load_dataset
    from hf_pipe import hf_pipe

    # 데이터셋 불러오기
    df = load_dataset("shangrilar/ko_text2sql", "origin")['test']
    df = df.to_pandas()
    for idx, row in df.iterrows():
        # make_prompt를 통해 LLM 추론에 사용할 프롬프트 생성
        prompt = make_prompt(row['context'], row['question'])
        df.loc[idx, 'prompt'] = prompt
    # 모델로 추론하여 SQL을 생성 후 저장
    gen_sqls = hf_pipe(df['prompt'].tolist(), do_sample=False,
                       return_full_text=False, max_length=512, truncation=True)
    gen_sqls = [x[0]['generated_text'] for x in gen_sqls]
    df['gen_sql'] = gen_sqls

    # 평가를 위한 requests.jsonl 생성
    eval_filepath = "text2sql_evaluation.jsonl"
    make_requests_for_gpt_evaluation(df, eval_filepath)
