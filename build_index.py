"""
build_index.py (v4)
Назначение: разбивка документов базы знаний на чанки,
           генерация эмбеддингов и создание FAISS индекса.

Изменения v4:
- Вся очистка вынесена в fetch_and_clean.py.
- build_index.py только чанкает и индексирует.
- Добавлена проверка размерности модели.

Требования:
    pip install sentence-transformers faiss-cpu langchain langchain-community tiktoken
"""

import json
import time
import pickle
from pathlib import Path
from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss

# ======================== КОНФИГУРАЦИЯ ========================

KNOWLEDGE_BASE_DIR = "knowledge_base"
INDEX_OUTPUT_DIR = "vector_index"
MODEL_NAME = "intfloat/multilingual-e5-small"

EXCLUDE_FILES = {"terms_map.json"}

CHUNK_SIZE = 460
CHUNK_OVERLAP = 80
MIN_CHUNK_LENGTH = 120

QUERY_INSTRUCTION = (
    "Instruct: Given a query, retrieve relevant passages that answer the query\nQuery: "
)
DOCUMENT_INSTRUCTION = (
    "Instruct: Represent the document for retrieval\nDocument: "
)


# ======================== ЗАГРУЗКА ДОКУМЕНТОВ ========================

def load_documents(base_dir: str, exclude: set) -> List[Dict]:
    """
    Загрузка всех .md файлов.
    Документы уже очищены на этапе fetch_and_clean.py.
    """
    documents = []
    base_path = Path(base_dir)

    if not base_path.exists():
        raise FileNotFoundError(f"Директория {base_dir} не найдена.")

    for filepath in sorted(base_path.glob("*.md")):
        if filepath.name in exclude:
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Извлечение заголовка
        title = filepath.stem.replace("_", " ")
        lines = content.split("\n")
        for line in lines:
            if line.strip().startswith("# "):
                title = line.strip()[2:].strip()
                break

        documents.append({
            "filepath": str(filepath),
            "filename": filepath.name,
            "title": title,
            "content": content,
        })

    return documents


# ======================== ЧАНКИНГ ========================

def chunk_documents(documents: List[Dict]) -> List[Dict]:
    """
    Разбивка документов на чанки.
    """
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks = []

    for doc in documents:
        chunks = text_splitter.split_text(doc["content"])

        for i, chunk_text in enumerate(chunks):
            chunk_text = chunk_text.strip()

            if len(chunk_text) < MIN_CHUNK_LENGTH:
                continue

            # Простая проверка: чанк должен содержать хотя бы 40% букв
            alpha_chars = sum(c.isalpha() for c in chunk_text)
            if alpha_chars / max(len(chunk_text), 1) < 0.4:
                continue

            all_chunks.append({
                "chunk_id": f"{doc['filename']}_chunk_{i:04d}",
                "title": doc["title"],
                "filename": doc["filename"],
                "filepath": doc["filepath"],
                "chunk_index": i,
                "text": chunk_text,
            })

    return all_chunks


# ======================== ГЕНЕРАЦИЯ ЭМБЕДДИНГОВ ========================

class EmbeddingGenerator:

    def __init__(self, model_name: str):
        print(f"Загрузка модели: {model_name}...")
        start = time.time()
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        elapsed = time.time() - start
        print(f"Модель загружена за {elapsed:.1f} сек.")
        self.dim = self.model.get_embedding_dimension()
        print(f"Размер эмбеддингов: {self.dim}")

        if self.dim != 384:
            raise RuntimeError(
                f"ОЖИДАЛАСЬ РАЗМЕРНОСТЬ 384, ПОЛУЧЕНА {self.dim}. "
                f"Проверьте MODEL_NAME и удалите кеш модели."
            )

    def encode_documents(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        instructed_texts = [DOCUMENT_INSTRUCTION + t for t in texts]
        embeddings = self.model.encode(
            instructed_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        return embeddings


# ======================== FAISS ИНДЕКС ========================

def build_faiss_index(embeddings: np.ndarray, chunks: List[Dict], output_dir: str):
    dim = embeddings.shape[1]
    num_vectors = embeddings.shape[0]

    print(f"Создание FAISS индекса: {num_vectors} векторов, размерность {dim}")

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(output_path / "knowledge_base.index"))
    print(f"Индекс сохранён: {output_path / 'knowledge_base.index'}")

    with open(output_path / "chunks_metadata.pkl", "wb") as f:
        pickle.dump(chunks, f)

    stats = {
        "model_name": MODEL_NAME,
        "embedding_dim": dim,
        "num_documents": len(set(c["filename"] for c in chunks)),
        "num_chunks": num_vectors,
        "chunk_size_tokens": CHUNK_SIZE,
        "chunk_overlap_tokens": CHUNK_OVERLAP,
        "min_chunk_length_chars": MIN_CHUNK_LENGTH,
        "index_type": "IndexFlatIP",
    }
    with open(output_path / "index_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


# ======================== ГЛАВНЫЙ ПАЙПЛАЙН ========================

def main():
    total_start = time.time()

    print("=" * 60)
    print("ЭТАП 1: Загрузка документов")
    documents = load_documents(KNOWLEDGE_BASE_DIR, EXCLUDE_FILES)
    print(f"Загружено документов: {len(documents)}")
    for doc in documents:
        print(f'  - {doc["filename"]}: "{doc["title"]}"')

    print("\n" + "=" * 60)
    print("ЭТАП 2: Разбивка на чанки")
    chunk_start = time.time()
    chunks = chunk_documents(documents)
    print(f"Создано чанков: {len(chunks)} за {time.time() - chunk_start:.1f} сек.")

    print("\n" + "=" * 60)
    print("ЭТАП 3: Генерация эмбеддингов")
    embedder = EmbeddingGenerator(MODEL_NAME)
    emb_start = time.time()
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode_documents(texts)
    print(f"Матрица: {embeddings.shape}, время: {time.time() - emb_start:.1f} сек.")

    print("\n" + "=" * 60)
    print("ЭТАП 4: Создание FAISS индекса")
    idx_start = time.time()
    build_faiss_index(embeddings, chunks, INDEX_OUTPUT_DIR)
    print(f"Время: {time.time() - idx_start:.1f} сек.")

    total_time = time.time() - total_start
    print(f"\nИТОГО: {len(chunks)} чанков, размерность {embedder.dim}, "
          f"время {total_time:.1f} сек.")


if __name__ == "__main__":
    main()