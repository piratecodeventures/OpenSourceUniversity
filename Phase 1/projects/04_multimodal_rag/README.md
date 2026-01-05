# 👁️‍🗨️ Multimodal RAG: Image Search Engine

> **Officer Kael's Log**: "Finding a needle in a haystack is hard. Finding a specific spaceship based on a vague pilot report ('It was red with spikes') is impossible... unless we map language and vision to the same space."

This project implements a **Dual Encoder** system inspired by **CLIP (Contrastive Language-Image Pre-Training)**.
It allows you to search for images using natural language queries.

## 🛠️ Architecture

### The Dual Encoder
1.  **Image Encoder**: ResNet50. Takes an Image $\to$ Vector (2048 dim).
2.  **Text Encoder**: DistilBERT. Takes Text $\to$ Vector (768 dim).
3.  **Projection Heads**: Linear layers that map both vectors to a shared **Latent Space** (256 dim).

### The Search (RAG)
1.  **Index**: We pass all images through the Image Encoder and store the vectors in a specialized Database (Vector DB).
2.  **Query**: User types "A red car".
3.  **Retrieval**: We pass text through Text Encoder.
4.  **Math**: Compute Cosine Similarity between Text Vector and all Image Vectors. Top matches are returned.

## 🚀 Usage

### 1. Install
```bash
pip install torch torchvision transformers pillow scikit-learn
```

### 2. Run the Engine
```bash
python search_engine.py
# 1. Initializes the Models.
# 2. Creates dummy images (Red Circle, Blue Square).
# 3. Embeds them.
# 4. Queries: "Find a red shape".
# 5. Returns: "red_circle.jpg" (High Score).
```

## 🧠 Key Concepts
*   **Contrastive Loss**: Training technique where we pull matching (Image, Text) pairs close and push mismatching pairs apart.
*   **Multimodal Embedding Space**: A mathematical space where the vector for "Dog" is close to the vector for an *image* of a dog.
