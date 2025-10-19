from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np

#Step 1 : Prepare Knowledge Base

docs =["PyTorch is an open-source machine learning framework based on Torch.",
    "Transformers library by Hugging Face provides pre-trained NLP models.",
    "Retrieval-Augmented Generation (RAG) combines search and generation.",
    "LoRA fine-tuning allows training large models efficiently with few parameters.",
    "Bitsandbytes enables quantization for efficient model inference."
]

#Step 2: Generate Embeddings

print("Loading SentenceTransformer model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("Model Loaded")

print("Generating document embeddings....")
doc_embeddings = embedder.encode(docs,convert_to_tensor = False)
doc_embeddings = np.array(doc_embeddings,dtype= 'float32')
print ("Embeddings generated for" ,len(docs),"documents.")

#Step 3 : Build FAISS Index

print("Building FAISS index...")
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(doc_embeddings)
print("FAISS index built with",index.ntotal,"documents.")

#Step 4: Define Retrieval Function

def retrieve (query, top_k = 2):
    q_emb = embedder.encode([query],convert_to_tensor= False)
    D,I = index.search(np.array(q_emb,dtype ='float32'),top_k)
    return [docs[i] for i in I[0]]

#Step 5: Load Generator(LLM)
print("Loading generation model (DistilGPT-2).. ")
generator = pipeline("text-generation",model ="distilgpt2")
print("Generator model ready!")

#Step 6 : RAG Chatbot Function

def rag_chatbot(query):
    retrieved_docs = retrieve(query)
    context = " ".join(retrieved_docs)
    prompt = f"Context:{context}\n\nQuestion :{query}\nAnswer:"
    response = generator(prompt,max_length=100,num_return_sequences=1,truncation=True)



    print("\n ======= ")
    print("Query :",query)
    print(" ========")
    print("\n --- Retrieved Context ---- ")
    for i, doc in enumerate(retrieved_docs,1):
        print(f"Doc {i}: {doc}")
    print ("\n-- LLM Response ---")
    print (response[0]['generated_text'])
    print("==============\n")


# Step 7:
if __name__ == "__main__":
    quries =["What is LoRA and why is it used?",
        "How does quantization help in model efficiency?",
        "What is RAG in NLP?"
    ]

    for q in quries :
        rag_chatbot(q)