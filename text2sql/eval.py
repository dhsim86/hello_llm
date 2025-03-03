from datasets import load_dataset

from prompt import make_prompt
from hf_pipe import hf_pipe

import json
from pathlib import Path


def make_requests_for_gpt_evaluation(df, filename, dir='requests'):
    if not Path(dir).exists():
        Path(dir).mkdir(parents=True)
    prompts = []
    for idx, row in df.iterrows():
        prompts.append(
            """Based on below DDL and Question, evaluate gen_sql can resolve Question. If gen_sql and gt_sql do equal job, return "yes" else return "no". Output JSON Format: {"resolve_yn": ""}""" + f"""

DDL: {row['context']}
Question: {row['question']}
gt_sql: {row['answer']}
gen_sql: {row['gen_sql']}"""
        )

    jobs = [{"model": "gpt-4-turbo-preview", "response_format": {"type": "json_object"},
             "messages": [{"role": "system", "content": prompt}]} for prompt in prompts]
    with open(Path(dir, filename), "w") as f:
        for job in jobs:
            json_string = json.dumps(job)
            f.write(json_string + "\n")


if __name__ == '__main__':
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
