import gc
import torch
import numpy as np
from datasets import Dataset
from transformers import AdamW


# GPU 메모리 사용량을 확인하는 함수 정의
def print_gpu_utilization():
    if torch.cuda.is_available():
        # torch.cuda.memory_allocated(): 현재 할당된 GPU 메모리의 양을 바이트 단위로 반환
        used_memory = torch.cuda.memory_allocated() / 1024 ** 3
        print(f"GPU 메모리 사용량: {used_memory:.3f} GB")
    elif torch.backends.mps.is_available():
        used_memory = torch.mps.current_allocated_memory() / 1024 ** 3
        print(f"GPU 메모리 사용량: {used_memory:.3f} GB")
    else:
        print("런타임 유형을 GPU로 변경해주세요.")


# 그레이디언트 메모리 사용량 계산
def estimate_memory_of_gradients(model):
    total_memory = 0
    for param in model.parameters():
        if param.grad is not None:
            # 모델에 저장된 그레이디언트 값의 수 * 값의 데이터 크기
            total_memory += param.grad.nelement() * param.grad.element_size()
    return total_memory


# 옵티마이저의 메모리 사용량 계산
def estimate_memory_of_optimizer(optimizer: AdamW):
    total_memory = 0
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                # 옵티마이저에 저장된 값의 수 * 값의 데이터 크기
                total_memory += v.nelement() * v.element_size()
    return total_memory


# 1. 전역 변수 중 GPU 메모리에 올라가는 모델의 변수(model) 및 데이터 셋 변수(dataset)을 삭제
# 2. 가비지 컬렉션을 통해 메모리 회수
# 3. GPU 메모리 반환
def clean_gpu_memory():
    if 'model' in globals():
        del globals()['model']
    if 'dataset' in globals():
        del globals()['dataset']
    gc.collect()

    # 더 이상 사용하지 않는 GPU 메모리 반환
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


# 랜덤 데이터셋 생성
# 텍스트의 길이가 256이고, 데이터가 64개인 랜덤 데이터셋 생성
def make_dummy_dataset():
    seq_len, dataset_size = 256, 64
    dummy_data = {
        'input_ids': np.random.randint(100, 30000, (dataset_size, seq_len)),
        'labels': np.random.randint(100, 30000, (dataset_size, seq_len))
    }

    dataset = Dataset.from_dict(dummy_data)
    dataset.set_format("pt")
    return dataset
