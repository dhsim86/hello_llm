import psutil


def get_memory_usage_mb():
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / (1024 * 1024)


import numpy as np

import time
import faiss
from faiss.contrib.datasets import DatasetSIFT1M

ds = DatasetSIFT1M()


def test_parameter_m():
    xq = ds.get_queries()
    xb = ds.get_database()
    gt = ds.get_groundtruth()

    k = 1
    d = xq.shape[1]
    nq = 1000
    xq = xq[:nq]

    # 반복문을 통해 파라미터 m을 8, 16, 32, 64로 키움
    for m in [8, 16, 32, 64]:
        # IndexHNSWFlat 메서드를 통해 HNSW 인덱스를 생성, m을 설정
        # 이때 ef_construction은 40, ef_search는 16으로 디폴트 값이다.
        index = faiss.IndexHNSWFlat(d, m)
        time.sleep(10)

        start_memory = get_memory_usage_mb()
        start_index = time.time()
        index.add(xb)  # 100만개의 임베딩 벡터를 색인
        end_memory = get_memory_usage_mb()
        end_index = time.time()
        print(f"M: {m} - 색인 시간: {end_index - start_index} s, 메모리 사용량: {end_memory - start_memory} MB")

        t0 = time.time()
        D, I = index.search(xq, k)
        t1 = time.time()

        recall_at_1 = np.equal(I, gt[:nq, :1]).sum() / float(nq)
        print(f"{(t1 - t0) * 1000.0 / nq:.3f} ms per query, R@1 {recall_at_1:.3f}")


def test_parameter_ef_consturction():
    xq = ds.get_queries()
    xb = ds.get_database()
    gt = ds.get_groundtruth()

    k = 1
    d = xq.shape[1]
    nq = 1000
    xq = xq[:nq]

    # ef_construction을 40, 80, 120, 160으로 키움
    for ef_construction in [40, 80, 160, 320]:
        # 파라미터 m은 32로 고정
        index = faiss.IndexHNSWFlat(d, 32)

        # ef_construction 파라미터는 index.hnsw.efConstruction 속성을 통해 설정
        index.hnsw.efConstruction = ef_construction
        time.sleep(3)

        start_memory = get_memory_usage_mb()
        start_index = time.time()
        index.add(xb)  # 100만개의 임베딩 벡터를 색인
        end_memory = get_memory_usage_mb()
        end_index = time.time()
        print(
            f"efConstruction: {ef_construction} - 색인 시간: {end_index - start_index} s, 메모리 사용량: {end_memory - start_memory} MB")

        t0 = time.time()
        D, I = index.search(xq, k)
        t1 = time.time()

        recall_at_1 = np.equal(I, gt[:nq, :1]).sum() / float(nq)
        print(f"{(t1 - t0) * 1000.0 / nq:.3f} ms per query, R@1 {recall_at_1:.3f}")


def test_parameter_ef_search():
    xq = ds.get_queries()
    xb = ds.get_database()
    gt = ds.get_groundtruth()

    k = 1
    d = xq.shape[1]
    nq = 1000
    xq = xq[:nq]

    # 파라미터 m은 32로 고정
    index = faiss.IndexHNSWFlat(d, 32)
    # 파라미터 ef_construction은 32로 고정
    index.hnsw.efConstruction = 320

    # 100만개의 임베딩 벡터를 색인
    index.add(xb)

    for ef_search in [16, 32, 64, 128]:
        index.hnsw.efSearch = ef_search
        t0 = time.time()
        D, I = index.search(xq, k)
        t1 = time.time()

        recall_at_1 = np.equal(I, gt[:nq, :1]).sum() / float(nq)
        print(f"{(t1 - t0) * 1000.0 / nq:.3f} ms per query, R@1 {recall_at_1:.3f}")


if __name__ == '__main__':
    test_parameter_ef_search()
