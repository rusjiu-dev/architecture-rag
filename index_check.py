import pickle
from pathlib import Path

with open('vector_index/chunks_metadata.pkl', 'rb') as f:
    chunks = pickle.load(f)

# Ищем чанки из malicious.md
malicious_chunks = [c for c in chunks if 'malicious' in c.get('filename', '').lower()]
print(f'Чанков из malicious.md: {len(malicious_chunks)}')
for c in malicious_chunks:
    chunk_id = c['chunk_id']
    text = c['text'][:200]
    print(f'  chunk_id: {chunk_id}')
    print(f'  text: {text}')
    print()

if not malicious_chunks:
    print('malicious.md НЕ проиндексирован!')