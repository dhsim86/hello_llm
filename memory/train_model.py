from datasets import Dataset
from transformers import TrainingArguments, AdamW
from torch.utils.data import DataLoader

from check_gpu_memory import print_gpu_utilization
from estimate_gradient_optimizer import estimate_memory_of_gradients, estimate_memory_of_optimizer


# 모델을 학습시키며 중간에 메모리 사용량 확인
def train_model(model, dataset: Dataset, training_args: TrainingArguments):
    # 그레이디언트 체크포인팅 사용 여부
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    train_dataloader = DataLoader(dataset, batch_size=training_args.per_device_train_batch_size)
    optimizer = AdamW(model.parameters())

    model.train()
    gpu_utilization_printed = False

    for step, batch in enumerate(train_dataloader, start=1):
        batch = {k: v.to(model.device) for k, v in batch.items()}

        outputs = model(**batch)

        loss = outputs.loss

        # TrainingArgument, 학습인자에서 그레이디언트 누적 횟수를 둘 수 있다.
        # >>>> 누적 횟수를 4로 둘 경우, 로스값(손실)을 4로 나누어 역전파를 수행
        loss = loss / training_args.gradient_accumulation_steps
        loss.backward()  # 손실값 역전파

        # 그레이디언트 누적 기능을 사용하는 설정
        # 스텝에 따라 2 또는 4로 설정하는데, 기본값은 1
        # >>>> 누적 횟수를 4로 둘 경우, 4번의 스텝마다 모델을 업데이트
        if step % training_args.gradient_accumulation_steps == 0:
            optimizer.step()  # 모델 업데이트 (역전파 결과를 바탕으로 모델 업데이트)

            # 그레이디언트 및 옵티마이저 메모리 사용량 계산
            gradients_memory = estimate_memory_of_gradients(model)
            optimizer_memory = estimate_memory_of_optimizer(optimizer)

            if not gpu_utilization_printed:
                # 전체 메모리 사용량
                print_gpu_utilization()
                gpu_utilization_printed = True

            optimizer.zero_grad()

            print(f"옵티마이저 상태의 메모리 사용량: {optimizer_memory / (1024 ** 3):.3f} GB")
            print(f"그레이디언트 상태의 메모리 사용량: {gradients_memory / (1024 ** 3):.3f} GB")
