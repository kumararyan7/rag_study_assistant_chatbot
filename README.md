from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss, numpy as np

# 1. Load model & data
embedder = SentenceTransformer('all-MiniLM-L6-v2')
docs = ["Deep learning enables RAG models.", "BERT is a transformer model."]
doc_embeddings = embedder.encode(docs, convert_to_tensor=False)

# 2. Create FAISS index
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(np.array(doc_embeddings))

# 3. Query + retrieve
query = "What is BERT?"
query_vector = embedder.encode([query])
_, indices = index.search(np.array(query_vector), k=1)
context = docs[indices[0][0]]

# 4. Generate response
generator = pipeline('text-generation', model='gpt2')
prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
print(generator(prompt, max_length=80, num_return_sequences=1))
