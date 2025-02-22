import gc
import torch


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
    torch.cuda.empty_cache()
