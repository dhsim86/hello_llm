if __name__ == '__main__':
    # 로컬의 csv 데이터 파일 사용
    from datasets import load_dataset
    dataset = load_dataset('csv', data_files='my_file.csv')

    # 파이썬 딕셔너리 사용
    from datasets import Dataset
    my_dict = {'a': [1, 2, 3]}
    dataset = Dataset.from_dict(my_dict)

    # 판다스 데이터프레임 활용
    import pandas as pd
    df = pd.DataFrame({"a": [1, 2, 3]})
    dataset = Dataset.from_pandas((df)
