#!/bin/bash

# OpenAI Encodings: https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken
# OpenAI RateLimits: https://platform.openai.com/docs/guides/rate-limits?tier=free#free-tier-rate-limits

export OPENAI_API_KEY=

REQUEST_PATH="requests/text2sql_evaluation.jsonl"
SAVE_PATH="results/text2sql_evaluation_result.jsonl"

# gpt-4o-mini ratelimit (Free Tier)
# - RPM: 3
# - TPM: 40,000

# cookbook의 소스(RateLimit를 관리하면서 jsonl에 담긴 요청을 전송하는 코드)를 실행
python utils/api_request_parallel_processor.py \
  --requests_filepath ${REQUEST_PATH} \
  --save_filepath ${SAVE_PATH} \
  --request_url https://api.openai.com/v1/chat/completions \
  --max_requests_per_minute 2 \
  --max_tokens_per_minute 30000 \
  --token_encoding_name o200k_base \
  --max_attempts 5 \
  --logging_level 20