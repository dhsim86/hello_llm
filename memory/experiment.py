import gc

import torch.cuda
from transformers import TrainingArguments, Trainer

from loader import load_model_and_tokenizer
from dummy_dataset import make_dummy_dataset
from train_model import train_model
from clean import clean_gpu_memory
from check_gpu_memory import print_gpu_utilization


# 배치 크기, 그레이디언트 누적, 그레이디언트 체크포인팅, peft 설정에 따라 GPU 메모리 사용량 확인
def gpu_memory_experiment(batch_size,
                          gradient_accumulation_steps=1,
                          gradient_checkpointing=False,
                          model_id="EleutherAI/polyglot-ko-1.3b",
                          peft=None):
    print(f"배치 크기: {batch_size}")

    # 모델 및 토크나이저, 데이터셋 로드
    model, tokenizer = load_model_and_tokenizer(model_id, peft=peft)
    dataset = make_dummy_dataset()

    if gradient_checkpointing == True or peft == 'qlora':
        model.config.use_cache = False

    # 학습 인자 설정
    training_args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        output_dir="./result",
        num_train_epochs=1
    )

    try:
        # 모델 학습하면서 GPU 메모리 사용량 확인
        train_model(model, dataset, training_args)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(e)
        else:
            raise e
    finally:
        # 메모리 정리, 모델과 데이터셋을 제거하고 GPU 메모리 반환
        del model, dataset
        gc.collect()
        torch.cuda.empty_cache()
        print_gpu_utilization()


if __name__ == '__main__':
    clean_gpu_memory()
    print_gpu_utilization()

    for batch_size in [4, 8, 16]:
        # 그레이디언트 누적
        # batch_size * gradient_accumulation_steps 크기 만큼의 배치 크기 효과를 얻을 수 있다.
        # gpu_memory_experiment(batch_size=batch_size, gradient_accumulation_steps=4)

        # 그레이디언트 체크포인팅
        # gpu_memory_experiment(batch_size=batch_size, gradient_accumulation_steps=1, gradient_checkpointing=True)

        # LoRA 적용
        gpu_memory_experiment(batch_size=batch_size, peft='lora')

        # QLoRA 적용
        # gpu_memory_experiment(batch_size=batch_size, peft='qlora')

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
