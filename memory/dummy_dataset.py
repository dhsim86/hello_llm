import numpy as np
from datasets import Dataset


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
