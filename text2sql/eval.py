if __name__ == '__main__':
    from datasets import load_dataset
    from utils.model import make_inference_pipeline
    from utils.prompt import make_prompt
    from utils.evaluation import make_requests_for_gpt_evaluation

    model_id = 'beomi/Yi-Ko-6B'
    model_pipe = make_inference_pipeline(model_id)

    # 데이터셋 불러오기
    df = load_dataset("shangrilar/ko_text2sql", "origin")['test']
    df = df.to_pandas()
    for idx, row in df.iterrows():
        # make_prompt를 통해 LLM 추론에 사용할 프롬프트 생성
        prompt = make_prompt(row['context'], row['question'])
        df.loc[idx, 'prompt'] = prompt
    # 모델로 추론하여 SQL을 생성 후 저장
    gen_sqls = model_pipe(df['prompt'].tolist(), do_sample=False,
                          return_full_text=False, max_length=512, truncation=True)
    gen_sqls = [x[0]['generated_text'] for x in gen_sqls]
    df['gen_sql'] = gen_sqls

    # 평가를 위한 requests.jsonl 생성
    eval_filepath = "text2sql_evaluation.jsonl"
    make_requests_for_gpt_evaluation(df, eval_filepath)
