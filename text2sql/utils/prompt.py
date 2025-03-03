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
