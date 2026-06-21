"""
rag_bot.py (v7 — bitsandbytes, Qwen2.5-0.5B, Few-shot + CoT)
- Двухпроходная проверка релевантности (Δ score).
- Few-shot: 3 примера (2 успешных + 1 "не знаю").
- Chain-of-Thought: пошаговые рассуждения в System Prompt и примерах.
"""

import pickle
import time
import json
import sys
from pathlib import Path
from typing import List, Dict

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# ======================== КОНФИГУРАЦИЯ ========================

INDEX_DIR = "vector_index"
EMBED_MODEL_NAME = "intfloat/multilingual-e5-small"
LLM_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
USE_4BIT = True

QUERY_INSTRUCTION = (
    "Instruct: Given a query, retrieve relevant passages that answer the query\nQuery: "
)

TOP_K_CHUNKS = 3
MAX_NEW_TOKENS = 300
DELTA_THRESHOLD = 0.006

SYSTEM_PROMPT = """You are a knowledge base assistant for a fictional sci-fi universe.
You must follow this process for every question:
1. Read the context documents carefully.
2. Identify which documents contain information relevant to the question.
3. Extract the specific facts that answer the question.
4. If the context contains the answer, write it clearly and cite the source.
5. If the context does NOT contain the answer, say: "This information is not available in the knowledge base."

Always show your reasoning as numbered steps before giving the final answer.
Always cite the source document name in brackets after the answer."""

FEW_SHOT_EXAMPLES = [
    {
        "query": "Who is Xarn Velgor?",
        "answer": (
            "Step 1: The question asks about Xarn Velgor's identity.\n"
            "Step 2: Document 'Darth Vader' states: 'Xarn Velgor was a legendary "
            "Flux-sensitive human male who was a Keeper Knight of the Stellar "
            "Concordium and the prophesied Chosen One of the Keepers of the Flux, "
            "destined to bring balance to the Synth Flux.'\n"
            "Step 3: This directly answers the question.\n"
            "Answer: Xarn Velgor was a legendary Flux-sensitive human male, a Keeper "
            "Knight of the Stellar Concordium, and the prophesied Chosen One of the "
            "Keepers of the Flux. [Source: Darth Vader]"
        )
    },
    {
        "query": "What is the Void Core?",
        "answer": (
            "Step 1: The question asks about the Void Core.\n"
            "Step 2: Document 'Death Star' describes: 'Void Core was a gargantuan "
            "space station armed with a planet-destroying superlaser powered by "
            "void crystals... originally designed by the Xarnak Hive.'\n"
            "Step 3: This provides a clear definition and names the builders.\n"
            "Answer: The Void Core was a gargantuan space station armed with a "
            "planet-destroying superlaser powered by void crystals, originally "
            "designed by the Xarnak Hive. [Source: Death Star]"
        )
    },
    {
        "query": "What is the capital city of the planet Xylophonia?",
        "answer": (
            "Step 1: The question asks about the capital city of Xylophonia.\n"
            "Step 2: None of the context documents mention Xylophonia or its capital.\n"
            "Step 3: The knowledge base does not contain this information.\n"
            "Answer: This information is not available in the knowledge base."
        )
    },
]


# ======================== LLM ========================

class BitsAndBytesLLM:
    def __init__(self, model_name=LLM_MODEL_NAME, use_4bit=USE_4BIT):
        print(f"Загрузка LLM: {model_name} (4-bit={use_4bit})...")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float32,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="cpu",
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        mem = self.model.get_memory_footprint() / 1e9
        print(f"      Память: {mem:.2f} GB")

    def generate(self, prompt: str, max_tokens=MAX_NEW_TOKENS) -> str:
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to("cpu") for k, v in inputs.items()}

        print(f"  Промпт: ~{len(prompt.split())} слов | Генерация...")

        t0 = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.1,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        gen_time = time.time() - t0

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        tps = len(generated_ids) / gen_time if gen_time > 0 else 0
        print(f"  Сгенерировано: {len(generated_ids)} ток. за {gen_time:.1f} сек. ({tps:.1f} ток/сек)")
        return response

    def generate_rag(self, query: str, context_chunks: List[Dict]) -> str:
        context_parts = []
        for i, chunk in enumerate(context_chunks[:2], 1):
            src = chunk.get("title", "Unknown")
            text = chunk.get("text", "")[:400]
            context_parts.append(f"[{src}]\n{text}")

        context_text = "\n\n".join(context_parts)

        # Формируем Few-shot примеры отдельно (без f-string с обратными слешами)
        example_1_q = FEW_SHOT_EXAMPLES[0]["query"]
        example_1_a = FEW_SHOT_EXAMPLES[0]["answer"]
        example_2_q = FEW_SHOT_EXAMPLES[1]["query"]
        example_2_a = FEW_SHOT_EXAMPLES[1]["answer"]
        example_3_q = FEW_SHOT_EXAMPLES[2]["query"]
        example_3_a = FEW_SHOT_EXAMPLES[2]["answer"]

        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Context:\n{context_text}\n\n"
            f"Examples of how to answer:\n\n"
            f"Q: {example_1_q}\nA: {example_1_a}\n\n"
            f"Q: {example_2_q}\nA: {example_2_a}\n\n"
            f"Q: {example_3_q}\nA: {example_3_a}\n\n"
            f"Now answer the following question using the same format.\n\n"
            f"Question: {query}"
        )
        return self.generate(prompt)


# ======================== RAG БОТ ========================

class RAGBot:
    def __init__(self):
        print("=" * 60)
        print("ИНИЦИАЛИЗАЦИЯ RAG-БОТА (Few-shot + CoT)")
        print("=" * 60)

        print("\n[1/3] FAISS индекс...")
        base = Path(INDEX_DIR)
        self.index = faiss.read_index(str(base / "knowledge_base.index"))
        with open(base / "chunks_metadata.pkl", "rb") as f:
            self.chunks = pickle.load(f)
        print(f"      {self.index.ntotal} векторов, d={self.index.d}")

        print(f"\n[2/3] Эмбеддер: {EMBED_MODEL_NAME}...")
        self.embed_model = SentenceTransformer(EMBED_MODEL_NAME, trust_remote_code=True)
        self.embed_dim = self.embed_model.get_embedding_dimension()
        print(f"      d={self.embed_dim}")

        if self.embed_dim != self.index.d:
            raise ValueError(f"Размерность: модель={self.embed_dim} != индекс={self.index.d}")

        print(f"\n[3/3] LLM: {LLM_MODEL_NAME}...")
        self.llm = BitsAndBytesLLM()

        print("\n" + "=" * 60)
        print("БОТ ГОТОВ К РАБОТЕ")
        print("=" * 60)

    def _get_best_score(self, query: str) -> float:
        instructed = QUERY_INSTRUCTION + query
        emb = self.embed_model.encode(
            [instructed], normalize_embeddings=True
        ).astype(np.float32)
        scores, _ = self.index.search(emb, 1)
        return float(scores[0][0])

    def _extract_main_entity(self, query: str) -> str:
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'who', 'what',
            'where', 'when', 'why', 'how', 'did', 'does', 'do', 'and',
            'or', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'its',
            'his', 'her', 'their', 'your', 'describe', 'explain',
            'relationship', 'connection', 'between', 'during',
            'role', 'play', 'conflict', 'philosophy', 'purpose',
            'capital', 'city', 'planet', 'invented', 'century',
            'about', 'from', 'by', 'into', 'over', 'under',
        }

        words = query.split()
        content_indices = []
        for i, w in enumerate(words):
            clean = w.strip('?!.,:;()[]{}"\'-')
            if clean.lower() not in stop_words and len(clean) > 2:
                content_indices.append(i)

        if not content_indices:
            return query

        entities = []
        current_entity = [words[content_indices[0]].strip('?!.,:;()[]{}"\'-')]

        for j, idx in enumerate(content_indices[1:], 1):
            prev_idx = content_indices[j - 1]
            if idx - prev_idx == 1:
                current_entity.append(words[idx].strip('?!.,:;()[]{}"\'-'))
            else:
                entities.append(' '.join(current_entity))
                current_entity = [words[idx].strip('?!.,:;()[]{}"\'-')]

        entities.append(' '.join(current_entity))
        return max(entities, key=len)

    def _is_relevant(self, query: str) -> tuple:
        full_score = self._get_best_score(query)
        main_entity = self._extract_main_entity(query)
        truncated_query = query.replace(main_entity, "").strip()
        truncated_query = ' '.join(truncated_query.split())

        if not truncated_query or truncated_query == query:
            return True, full_score, full_score

        truncated_score = self._get_best_score(truncated_query)
        delta = full_score - truncated_score
        is_relevant = delta >= DELTA_THRESHOLD
        return is_relevant, full_score, truncated_score

    def search(self, query: str) -> List[Dict]:
        instructed = QUERY_INSTRUCTION + query
        query_emb = self.embed_model.encode(
            [instructed], normalize_embeddings=True
        ).astype(np.float32)

        scores, indices = self.index.search(query_emb, TOP_K_CHUNKS)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(score)
            results.append(chunk)
        return results

    def ask(self, query: str, verbose: bool = True) -> Dict:
        if verbose:
            print(f"\n{'─'*60}")
            print(f"ЗАПРОС: {query}")
            print(f"{'─'*60}")

        is_relevant, full_score, truncated_score = self._is_relevant(query)

        if verbose:
            print(f"  Score полного запроса:    {full_score:.4f}")
            print(f"  Score без сущности:       {truncated_score:.4f}")
            print(f"  Разница (Δ):              {full_score - truncated_score:.4f}")
            print(f"  Порог (≥ {DELTA_THRESHOLD}):         {'РЕЛЕВАНТЕН' if is_relevant else 'НЕРЕЛЕВАНТЕН'}")

        if not is_relevant:
            answer = (
                "Step 1: The question asks about information not found in the context.\n"
                "Step 2: No relevant documents were identified.\n"
                "Answer: This information is not available in the knowledge base."
            )
            if verbose:
                print(f"\nОТВЕТ:\n{answer}")
            return {
                "query": query, "answer": answer,
                "context_chunks": [], "search_time": 0.0,
                "gen_time": 0.0, "is_unknown": True,
                "full_score": full_score,
                "delta": full_score - truncated_score,
            }

        t0 = time.time()
        chunks = self.search(query)
        search_time = time.time() - t0

        if verbose:
            print(f"  Чанков найдено: {len(chunks)} за {search_time:.3f} сек.")

        t0 = time.time()
        answer = self.llm.generate_rag(query, chunks)
        gen_time = time.time() - t0

        is_unknown = any(
            phrase in answer.lower()
            for phrase in [
                "not available in the knowledge base",
                "not found in the context",
            ]
        )

        if verbose:
            print(f"\nОТВЕТ (поиск: {search_time:.3f}с | генерация: {gen_time:.1f}с):")
            print(answer)

        return {
            "query": query, "answer": answer,
            "context_chunks": chunks, "search_time": search_time,
            "gen_time": gen_time, "is_unknown": is_unknown,
            "full_score": full_score,
            "delta": full_score - truncated_score,
        }


# ======================== ТЕСТЫ ========================

def run_tests():
    print("\n" + "#" * 60)
    print("ТЕСТОВЫЕ ДИАЛОГИ (Few-shot + Chain-of-Thought)")
    print("#" * 60)

    bot = RAGBot()

    queries = [
        "Who is Xarn Velgor and what is his connection to the Synth Flux?",
        "What is the Void Core and who built it?",
        "Describe the philosophy of the Keepers of the Flux and their conflict with the Shade Covenant.",
        # Запросы без ответа
        "What is the capital city of the planet Xylophonia?",
        "Who invented the hyperdrive engine in the 23rd century?",
    ]

    results = []
    for i, q in enumerate(queries, 1):
        print(f"\n{'#'*60}")
        print(f"ТЕСТ {i}/{len(queries)}")
        print(f"{'#'*60}")
        result = bot.ask(q, verbose=True)
        results.append(result)

    known = [r for r in results if not r["is_unknown"]]
    unknown = [r for r in results if r["is_unknown"]]

    print(f"\n{'='*60}")
    print(f"ИТОГО: успешных={len(known)}, 'не знаю'={len(unknown)}")
    if known:
        print(f"Среднее время генерации: {np.mean([r['gen_time'] for r in known]):.1f} сек.")

    with open("test_dialogs.json", "w", encoding="utf-8") as f:
        json.dump([
            {
                "query": r["query"],
                "answer": r["answer"],
                "is_unknown": r["is_unknown"],
                "full_score": round(r.get("full_score", 0), 4),
                "delta": round(r.get("delta", 0), 4),
            }
            for r in results
        ], f, indent=2, ensure_ascii=False)

    print(f"Сохранено: test_dialogs.json")


def interactive():
    bot = RAGBot()
    print("\nВведите запрос (exit — выход):\n")
    while True:
        try:
            q = input("> ").strip()
            if q.lower() in ("exit", "quit", "q"):
                break
            if q:
                bot.ask(q, verbose=True)
        except (EOFError, KeyboardInterrupt):
            break


if __name__ == "__main__":
    if "--test" in sys.argv:
        run_tests()
    else:
        interactive()