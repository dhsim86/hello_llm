import psutil


def get_memory_usage_mb():
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / (1024 * 1024)


import time
import faiss
from faiss.contrib.datasets import DatasetSIFT1M

ds = DatasetSIFT1M()

xq = ds.get_queries()
xb = ds.get_database()
gt = ds.get_groundtruth()

k = 1
d = xq.shape[1]
nq = 1000
xq = xq[:nq]

start_memory = get_memory_usage_mb()
for i in range(1, 10, 2):
    start_indexing = time.time()

    index = faiss.IndexFlatL2(d)
    index.add(xb[:(i + 1) * 100000])

    end_indexing = time.time()
    end_memory = get_memory_usage_mb()

    t0 = time.time()
    D, I = index.search(xq, k)
    t1 = time.time()

    print(f"데이터 {(i + 1) * 100000}개:")
    print(
        f"색인: {(end_indexing - start_indexing) * 1000 :.3f} ms ({end_memory - start_memory:.3f} MB) 검색: {(t1 - t0) * 1000 / nq :.3f} ms")
