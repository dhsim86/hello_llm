from datasets import load_dataset
import numpy as np

klue_mrc_test = load_dataset('klue', 'mrc', split='validation')
klue_mrc_test = klue_mrc_test.train_test_split(test_size=1000, seed=42)['test']

import faiss


def make_embedding_index(sentence_model, corpus):
    embeddings = sentence_model.encode(corpus)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index


def find_embedding_top_k(query, sentence_model, index, k):
    embedding = sentence_model.encode([query])
    distances, indices = index.search(embedding, k)
    return indices


def make_question_context_pairs(question_idx, indices):
    return [[klue_mrc_test['question'][question_idx], klue_mrc_test['context'][idx]] for idx in indices]


def rerank_top_k(cross_model, question_idx, indices, k):
    input_examples = make_question_context_pairs(question_idx, indices)
    relevance_scores = cross_model.predict(input_examples)
    reranked_indices = indices[np.argsort(relevance_scores)[::-1]]
    return reranked_indices


import time


def evaluate_hit_rate(datasets, embedding_model, index, k=10):
    start_time = time.time()
    predictions = []
    for question in datasets['question']:
        predictions.append(find_embedding_top_k(question, embedding_model, index, k)[0])
    total_prediction_count = len(predictions)
    hit_count = 0
    questions = datasets['question']
    contexts = datasets['context']
    for idx, prediction in enumerate(predictions):
        for pred in prediction:
            if contexts[pred] == contexts[idx]:
                hit_count += 1
                break
    end_time = time.time()
    return hit_count / total_prediction_count, end_time - start_time


from sentence_transformers import SentenceTransformer

base_embedding_model = SentenceTransformer('raveas/klue-roberta-base-klue-sts')
base_index = make_embedding_index(base_embedding_model, klue_mrc_test['context'])
print("base embedding model hit rate")
print(evaluate_hit_rate(klue_mrc_test, base_embedding_model, base_index, 10))

finetuned_embedding_model = SentenceTransformer('raveas/klue-roberta-base-klue-sts-mrc')
finetuned_index = make_embedding_index(finetuned_embedding_model, klue_mrc_test['context'])
print("finetuned embedding model hit rate")
print(evaluate_hit_rate(klue_mrc_test, finetuned_embedding_model, finetuned_index, 10))

import time
from tqdm.auto import tqdm


def evaluate_hit_rate_with_rerank(datasets, embedding_model, cross_model, index, bi_k=30, cross_k=10):
    start_time = time.time()
    predictions = []
    for question_idx, question in enumerate(tqdm(datasets['question'])):
        indices = find_embedding_top_k(question, embedding_model, index, bi_k)[0]
        predictions.append(rerank_top_k(cross_model, question_idx, indices, k=cross_k))
    total_prediction_count = len(predictions)
    hit_count = 0
    questions = datasets['question']
    contexts = datasets['context']
    for idx, prediction in enumerate(predictions):
        for pred in prediction:
            if contexts[pred] == contexts[idx]:
                hit_count += 1
                break
    end_time = time.time()
    return hit_count / total_prediction_count, end_time - start_time, predictions


cross_model = SentenceTransformer('shangrilar/klue-roberta-small-cross-encoder')
hit_rate, cosumed_time, predictions = evaluate_hit_rate_with_rerank(klue_mrc_test,
                                                                    finetuned_embedding_model,
                                                                    cross_model,
                                                                    finetuned_index,
                                                                    bi_k=30,
                                                                    cross_k=10)
print(hit_rate, cosumed_time)
