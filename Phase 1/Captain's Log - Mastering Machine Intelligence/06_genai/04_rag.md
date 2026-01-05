# Generative AI: RAG (Retrieval-Augmented Generation)

## 📜 Story Mode: The Open Book Exam

> **Mission Date**: 2043.10.15
> **Location**: Deep Space Outpost "Vector Prime"
> **Officer**: Lead Engineer Kael
>
> **The Problem**: The LLM is smart, but its knowledge Cut-Off was 2042.
> It doesn't know about the "Asteroid Event" from yesterday.
> If I ask "What is the shield status?", it hallucinates.
>
> I can't retrain it every day. Too expensive ($10M).
>
> **The Solution**: Give it a textbook.
> I will **Retrieve** the relevant manual pages about "Shields".
> I will paste them into the prompt.
> "Context: Shields are at 40%. Question: What is shield status?"
>
> *"Computer! Index the Ship's Logs into the Vector Database. Connect the Retriever to the Generator."*

---

## 1. Problem Setup & Motivation

### The 6 Engineering Questions
1.  **WHAT**:
    *   **RAG**: combining a Retriever (Search Engine) with a Generator (LLM).
    *   **Vector DB**: A database optimized for similarity search (Embeddings).
2.  **WHY**: LLMs have stale knowledge and hallucinate facts. RAG grounds them in real data.
3.  **WHEN**: Enterprise Search, Customer Support chatbots using company PDFs.
4.  **WHERE**: `Pinecone`, `Milvus`, `ChromaDB`, `LangChain`.
5.  **WHO**: Lewis et al. (Facebook AI, 2020).
6.  **HOW**: `Query -> Embed -> Search VectorDB -> Top K Chunks -> Prompt LLM`.

> [!NOTE]
> **🛑 Pause & Explain (In Simple Words)**
>
> **The Open Book Exam.**
>
> - **Pure LLM**: Taking a test from memory. You might make up facts.
> - **RAG**: You are allowed to run to the library, photocopy 3 pages relevant to the question, and keep them on your desk while answering.

---

## 2. Mathematical Problem Formulation

### Cosine Similarity Search
Given Query Vector $Q$ and Document Vector $D$:
$$ \text{Sim}(Q, D) = \frac{Q \cdot D}{||Q|| ||D||} $$
We want $\text{argmax}_{D \in \text{Database}} \text{Sim}(Q, D)$.
For 1 Billion vectors, we use **HNSW** (Hierarchical Navigable Small World) graphs to find the approximate neighbor in $O(\log N)$.

---

## 3. The Trifecta: Implementation Levels

We will implement a **Vector Search**.

### The Ship's Code (Polyglot: Pure Python + Libraries)

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# LEVEL 0: Pure Python (Linear Scan)
# O(N) - Slow but exact
def vector_search_pure(query_vec, db_vectors):
    scores = []
    for doc_vec in db_vectors:
        # Dot product (assuming normalized)
        score = sum(q * d for q, d in zip(query_vec, doc_vec))
        scores.append(score)
    
    # Get index of best score
    best_idx = scores.index(max(scores))
    return best_idx, max(scores)

# LEVEL 1: NumPy (Matrix Multiply)
# Fast for small N (<100k)
def vector_search_numpy(query_vec, db_matrix):
    # query: (1, D)
    # db: (N, D)
    # scores: (1, N)
    scores = np.dot(db_matrix, query_vec)
    best_idx = np.argmax(scores)
    return best_idx, scores[best_idx]

# LEVEL 2: Faiss (Production)
# Facebook AI Similarity Search
# Uses IndexFlatIP (Inner Product)
"""
import faiss
index = faiss.IndexFlatIP(d) 
index.add(db_matrix)
D, I = index.search(query_vec, k=5)
"""
"""
```

> [!TIP]
> **👁️ Visualizing the Search: The Needle in the Haystack**
> Run this script to see how the Query Vector finds its neighbors.
>
> ```python
> import matplotlib.pyplot as plt
> import numpy as np
>
> def plot_vector_search():
>     # 1. Generate Random Database (2D for Viz)
>     np.random.seed(42)
>     db_vectors = np.random.randn(20, 2)
>     
>     # Normalize to unit length (Cosine Similarity geometry)
>     db_vectors = db_vectors / np.linalg.norm(db_vectors, axis=1, keepdims=True)
>     
>     # 2. Generate Query
>     query = np.array([1, 0.5])
>     query = query / np.linalg.norm(query)
>     
>     # 3. Calculate Similarity
>     scores = np.dot(db_vectors, query)
>     top_k_idx = np.argsort(scores)[::-1][:3] # Top 3
>     
>     # 4. Plot
>     plt.figure(figsize=(8, 8))
>     origin = np.array([0, 0])
>     
>     # Plot DB (Gray)
>     plt.quiver(np.zeros(20), np.zeros(20), db_vectors[:,0], db_vectors[:,1], 
>                color='gray', alpha=0.3, scale=1, scale_units='xy', angles='xy')
>                
>     # Plot Query (Blue)
>     plt.quiver(0, 0, query[0], query[1], color='blue', scale=1, scale_units='xy', angles='xy', label='Query', linewidth=2)
>     
>     # Plot Top K (Green)
>     for i in top_k_idx:
>         vec = db_vectors[i]
>         plt.quiver(0, 0, vec[0], vec[1], color='green', scale=1, scale_units='xy', angles='xy', linewidth=2)
>         plt.text(vec[0]*1.1, vec[1]*1.1, "Hit", color='green')
>         
>     plt.xlim(-1.5, 1.5)
>     plt.ylim(-1.5, 1.5)
>     plt.grid(True)
>     plt.gca().set_aspect('equal')
>     plt.title(f"Vector Search (Cosine Similarity)\nQuerying the Vector Database")
>     plt.legend()
>     plt.show()
>
> # Uncomment to run:
> # plot_vector_search()
> ```

---

## 4. System-Level Integration

```mermaid
graph LR
    User[Query] --> Embed[Embedding Model]
    Embed --> Vec[Vector]
    Vec --> VDB[Vector DB (Pinecone)]
    VDB -- Top K Chunks --> Prompt
    Prompt --> LLM
    LLM --> Answer
```

**Where it lives**:
**Bing Chat**: RAG over the generic internet search results.

---

## 5. Evaluation & Failure Analysis

### Failure Mode: Retrieval mismatch
Query: "How to kill a Python process?"
Retrieved: "Python is a snake living in Africa." (Semantic match on "Python").
**Fix**: Hybrid Search (Keywords "Process" + Semantic "Kill/Terminate").

---

## 13. Assessment & Mastery Checks

**Q1: Chunking**
Why do we split text into chunks (e.g., 512 tokens)?
*   *Answer*: Vectors represent *one* concept. If you embed a whole book into one vector, the meaning is diluted. Also, LLM context windows are limited.

**Q2: Embeddings**
Do I use OpenAI embeddings or generic BERT?
*   *Answer*: Use models trained for *Retrieval* (e.g., `text-embedding-3`). Generic BERT is for classification.

**Q3: Latency**
RAG is slow. Why?
*   *Answer*: You have 3 steps: Embedding (20ms) + Search (50ms) + Generation (2000ms). Generation is the bottleneck.

### 14. Common Misconceptions (Debug Your Thinking)

> [!WARNING]
> **"RAG teaches the model new things."**
> *   **Correction**: No. The model learns nothing. It just "reads" the new info for that one turn. If you delete the Vector DB, it forgets.

> [!TIP]
> **RAG Architecture**
> ```mermaid
> graph TD
>     Doc[PDF Document] --> Chunker[Chunker (Split Text)]
>     Chunker --> EmbeddingModel
>     EmbeddingModel --> VectorDB[Vector Database]
>     
>     User[User Query] --> EmbeddingModel
>     EmbeddingModel --> SearchVector[Query Vector]
>     
>     SearchVector -- "Similarity Search" --> VectorDB
>     VectorDB -- "Top K Chunks" --> Context
>     
>     Prompt[Prompt] -- "Format" --> FinalPrompt
>     User -- "Question" --> FinalPrompt
>     Context -- "Knowledge" --> FinalPrompt
>     
>     FinalPrompt --> LLM
>     LLM --> Answer
>     
>     style VectorDB fill:#bbf,stroke:#333
>     style LLM fill:#bfb,stroke:#333
> ```

> [!WARNING]
> **"Vector DB is magic."**
> *   **Correction**: It's just math. If your embeddings are bad (Garbage In), your search results are bad (Garbage Out).
