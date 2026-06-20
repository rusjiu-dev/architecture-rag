import pickle
import time
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

INDEX_DIR = "vector_index"
MODEL_NAME = "intfloat/multilingual-e5-small"  # Должна совпадать с build_index.py!   intfloat/multilingual-e5-small
QUERY_INSTRUCTION = "Instruct: Given a query, retrieve relevant passages that answer the query\nQuery: "

def load_index(index_dir: str):
    base = Path(index_dir)
    index = faiss.read_index(str(base / "knowledge_base.index"))
    with open(base / "chunks_metadata.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def search(query: str, index, chunks, model, top_k: int = 5):
    instructed = QUERY_INSTRUCTION + query
    query_emb = model.encode(
        [instructed],
        normalize_embeddings=True,
    ).astype(np.float32)

    start = time.time()
    scores, indices = index.search(query_emb, top_k)
    elapsed = time.time() - start

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = chunks[idx].copy()
        chunk["score"] = float(score)
        results.append(chunk)

    return results, elapsed

def main():
    print("Загрузка индекса и модели...")
    index, chunks = load_index(INDEX_DIR)
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
    print(f"Индекс: {index.ntotal} векторов, размерность {index.d}")

    test_queries = [
        "Who is Xarn Velgor and what is his connection to the Synth Flux?",
        "What is the Void Core and who built it?",
        "Describe the philosophy of the Keepers of the Flux and their conflict with the Shade Covenant."
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"ЗАПРОС {i}: {query}")
        print(f"{'='*60}")

        results, elapsed = search(query, index, chunks, model, top_k=3)

        print(f"Время поиска: {elapsed:.4f} сек.")
        print(f"Найдено результатов: {len(results)}\n")

        for j, r in enumerate(results, 1):
            print(f"--- Результат {j} (score: {r['score']:.4f}) ---")
            print(f"Источник: {r['title']}")
            print(f"Файл:     {r['filename']}")
            print(f"Чанк:     {r['chunk_id']}")
            print(f"Текст:\n{r['text'][:500]}...")
            print()

if __name__ == "__main__":
    main()