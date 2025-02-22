import torch


# GPU 메모리 사용량을 확인하는 함수 정의
def print_gpu_utilization():
    if torch.cuda.is_available():
        # torch.cuda.memory_allocated(): 현재 할당된 GPU 메모리의 양을 바이트 단위로 반환
        used_memory = torch.cuda.memory_allocated() / 1024 ** 3
        print(f"GPU 메모리 사용량: {used_memory:.3f} GB")
    else:
        print("런타임 유형을 GPU로 변경해주세요.")


if __name__ == '__main__':
    print_gpu_utilization()
