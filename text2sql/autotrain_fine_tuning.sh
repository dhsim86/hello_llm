#!/bin/bash

# 기초 모델 이름
BASE_MODEL='beomi/Yi-Ko-6B'
# 미세 조정된 모델의 이름
FINETUNED_MODEL='yi-ko-6b-text2sql'

# 데이터 경로(data/) 및 사용할 컬럼 (프롬프트가 저장된 text 컬럼) 지정
DATA_PATH='data/'
TEXT_COLUMN='text'

# --use-peft 파라미터 변경됨
# https://github.com/huggingface/autotrain-advanced/issues/415#issuecomment-1863861915

# macOS에서 불가능
# --mixed-precision fp16 \
# --quantization int4 \

autotrain llm \
--train \
--model ${BASE_MODEL} \
--project-name ${FINETUNED_MODEL} \
--data-path ${DATA_PATH} \
--text-column ${TEXT_COLUMN} \
--lr 2e-4 \
--batch-size 8 \
--epochs 1 \
--block-size 1024 \
--warmup-ratio 0.1 \
--lora-r 16 \
--lora-alpha 32 \
--lora-dropout 0.05 \
--weight-decay 0.01 \
--gradient-accumulation 8 \
--peft \
--quantization None \
--trainer sft
