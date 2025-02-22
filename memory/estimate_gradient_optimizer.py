import torch
from transformers import AdamW


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
