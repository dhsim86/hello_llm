from datasets import load_dataset
from prompt import make_prompt

# 학습 데이터 다운로드
df_sql = load_dataset("shangrilar/ko_text2sql", "origin")["train"]
df_sql = df_sql.to_pandas()
df_sql = df_sql.dropna().sample(frac=1, random_state=42)
# 평가에 사용할 게임 카테고리(db_id == 1)은 제거
df_sql = df_sql.query("db_id != 1")

for idx, row in df_sql.iterrows():
    # 프롬프트 생성 후 'text' 컬럼에 저장
    df_sql.loc[idx, 'text'] = make_prompt(row['context'], row['question'], row['answer'])

# data 폴더에 저장
df_sql.to_csv('data/train.csv', index=False)