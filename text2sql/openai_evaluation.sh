#!/bin/bash

# OpenAI Encodings: https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken
# OpenAI RateLimits: https://platform.openai.com/docs/guides/rate-limits?tier=free#free-tier-rate-limits

export OPENAI_API_KEY=sk-1234567890abcdef123456789

# gpt-4o-mini ratelimit
# - RPM: 3
# - TPM: 40,000

# cookbook의 소스(RateLimit를 관리하면서 jsonl에 담긴 요청을 전송하는 코드)를 실행
python api_request_parallel_processor.py \
  --requests_filepath {요청 파일 경로} \
  --save_filepath {생성할 결과 파일 경로} \
  --request_url https://api.openai.com/v1/chat/completions \
  --max_requests_per_minute 2 \
  --max_tokens_per_minute 30000 \
  --token_encoding_name o200k_base \
  --max_attempts 5 \
  --logging_level 20python api_re