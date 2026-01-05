# Path A: GenAI & LLMs Specialist (The Architect)

## 📜 Career Profile: The Architect
*   **Role**: AI Research Scientist, LLM Engineer, Generative AI Specialist.
*   **Mission**: You don't just *call* APIs. You *build* the models that others call. You understand the "Ghost in the Machine".
*   **Income Potential**: $200k - $500k+ (Equity heavy).
*   **Core Stack**: PyTorch, CUDA, HuggingFace, vLLM, LangChain.

---

## 📅 Sprint 11: Modern Generative Models (Weeks 41-44)

> **Theme**: "Pixels and Probabilities"

### 11.1 Diffusion Models (The Physics of Creation)
*   **11.1.1 The Forward Process (Destruction)**:
    *   Adding Gaussian noise $q(x_t|x_{t-1})$ until the image is pure static $\mathcal{N}(0, I)$.
    *   *Math*: $\alpha_t = 1 - \beta_t$. The "Signal to Noise Ratio".
*   **11.1.2 The Reverse Process (Creation)**:
    *   Training a U-Net to predict the noise $\epsilon_\theta(x_t, t)$.
    *   *Algorithm*: DDPM (Denoising Diffusion Probabilistic Models).
    *   *Sampling*: DDIM (Deterministic sampling for speed).
*   **11.1.3 Stable Diffusion (Latent Space)**:
    *   Why pixel space is slow ($256 \times 256 \times 3$).
    *   Using **VQ-VAE** to compress image to latent $z$ ($64 \times 64 \times 4$).
    *   Conditioning with **CLIP** text embeddings (Cross-Attention).

### 11.2 Advanced GANs (Adversarial Mastery)
*   **11.2.1 StyleGAN (The Standard for Resolution)**:
    *   **Mapping Network**: $z \to w$ (Disentangled Latent Space).
    *   **AdaIN (Adaptive Instance Norm)**: Injecting style $w$ into Conv layers.
    *   **Mixing Regularization**: Using different $w$ codes for coarse/fine layers.
*   **11.2.2 CycleGAN (Unpaired Translation)**:
    *   *Goal*: Horse $\to$ Zebra without pairs.
    *   *Cycle Consistency Loss*: $F(G(x)) \approx x$.
    *   *Identity Loss*: If you feed a Zebra to "Horse2Zebra", it should stay a Zebra.

### 11.3 Variational Autoencoders (VAEs)
*   **11.3.1 The ELBO**: Evidence Lower Bound optimization.
*   **11.3.2 VQ-VAE (Vector Quantized)**:
    *   Discrete Codebook learning.
    *   Avoids "blurry" VAE outputs by snapping to nearest code vector.
    *   Crucial for GenAI (DALL-E 1 tokenized images this way).

---

## 📅 Sprint 12: LLM Engineering Deep Dive (Weeks 45-48)

> **Theme**: "Taming the Beast"

### 12.1 Transformer Architectures (Beyond Attention)
*   **12.1.1 Positional Embeddings**:
    *   **RoPE (Rotary Positional Embeddings)**: Rotating the vector in complex space. Better extrapolation.
    *   **ALiBi**: Linear bias for length extrapolation.
*   **12.1.2 Attention Variants**:
    *   **MQA (Multi-Query Attention)**: Share Key/Value heads.
    *   **GQA (Grouped-Query Attention)**: The Llama-2/3 standard (Balance speed/quality).
    *   **FlashAttention**: IO-aware exact attention (Tiling to avoid HBM reads).

### 12.2 Distributed Training (Scale)
*   **12.2.1 Data Parallelism (DDP)**: Replicate model, split data. Sync Gradients.
*   **12.2.2 FSDP (Fully Sharded Data Parallel)**:
    *   Shard *Optimizer State*, *Gradients*, and *Parameters* across GPUs.
    *   Allows training 70B models on smaller clusters.
*   **12.2.3 3D Parallelism**: Combining Data + Tensor + Pipeline parallelism (Megatron-LM).

### 12.3 Fine-Tuning Strategies
*   **12.3.1 Instruction Tuning (SFT)**:
    *   Formatting data: `{"role": "user", "content": ...}`.
    *   Chat Templates (`<|im_start|>...`).
*   **12.3.2 Parameter Efficient (PEFT)**:
    *   **LoRA**: Matrix Rank decomposition. $W_{new} = W + \frac{\alpha}{r}AB$.
    *   **QLoRA**: 4-bit Quantization + Paged Optimizers (Training on 1 GPU).

---

## 📅 Sprint 13: Agents & Multi-Modal Systems (Weeks 49-52)

> **Theme**: "From Chatbot to Employee"

### 13.1 Reasoning Engines
*   **13.1.1 Chain of Thought (CoT)**: "Let's think step by step".
*   **13.1.2 ReAct (Reason + Act)**:
    *   Loop: Thought $\to$ Action (Tool Call) $\to$ Observation (Result) $\to$ Thought.
*   **13.1.3 Tree of Thoughts (ToT)**: DFS/BFS search over possible reasoning paths.

### 13.2 Tool Use & Function Calling
*   **13.2.1 Grammar Constrained Generation**:
    *   Forcing the LLM to output valid JSON.
    *   `GBNF` grammars in Llama.cpp.
*   **13.2.2 Tool Retrieval**: Finding the right tool from a library of 1000 tools (Retriever logic).

### 13.3 Multi-Modal LLMs (VLMs)
*   **13.3.1 Connectivity**:
    *   **Visual Encoder** (CLIP/SigLIP) $\to$ **Projector** (MLP) $\to$ **LLM Embedding Space**.
    *   "The image is just a foreign language token".
*   **13.3.2 LLaVA Architecture**: Visual Instruction Tuning.

---

## 📅 Sprint 14: Capstone - The Enterprise Brain (Weeks 53-56)

### 14.1 Enterprise RAG Architecture
*   **14.1.1 Advanced Retrieval**:
    *   **Hybrid Search**: Dense Vector (Semantic) + Sparse BM25 (Keyword).
    *   **Re-Ranking**: Cross-Encoders (Cohere/BGE) to filter top-k.
    *   **Query Expansion**: Hypothetical Document Embeddings (HyDE).
*   **14.1.2 Evaluation (RAGAS)**:
    *   **Faithfulness**: Does answer match context?
    *   **Answer Relevance**: Does answer match query?
    *   **Context Precision**: Was the right chunk retrieved?

### 14.2 Production Serving
*   **14.2.1 vLLM & PagedAttention**:
    *   KV-Cache memory management.
    *   Continuous Batching (No waiting for other requests to finish).
*   **14.2.2 Speculative Decoding**:
    *   Draft model (Small) guesses tokens, Main model (Large) verifies.

### 14.3 Safety & Guardrails
*   **14.3.1 Input Rails**: PII Redaction, Jailbreak detection (LlamaGuard).
*   **14.3.2 Output Rails**: Hallucination checks, Tone policing.

---

## 💻 The Ship's Code: Unsloth QLoRA Training

```python
from unsloth import FastLanguageModel
import torch

def production_finetune():
    # 1. Load 4-bit Model (Memory Efficient)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/llama-3-8b-bnb-4bit",
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
    )

    # 2. Add LoRA Adapters (Targeting all linear layers)
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Rank
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0, # Optimized to 0
        bias = "none",
        use_gradient_checkpointing = True,
    )

    # 3. Data Formatting (Alpaca/ChatML)
    alpaca_prompt = """Below is an instruction...
    ### Instruction:
    {}
    ### Input:
    {}
    ### Response:
    {}"""
    
    # 4. Train
    trainer = SFTTrainer(
        model = model,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = 2048,
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            max_steps = 60,
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            optim = "adamw_8bit",
        ),
    )
    trainer.train()
```

---

## ❓ Industry Interview Corner

**Q1: "Explain the difference between Post-Norm and Pre-Norm in Transformers."**
*   **Answer**: "Original Transformer used Post-Norm (LayerNorm *after* Attention/MLP). This was hard to train (gradient explosion). Modern LLMs (Llama, GPT-3) use **Pre-Norm** (LayerNorm *before* blocks). This stabilizes gradients but slightly limits representation power."

**Q2: "What is the Chinchilla Scaling Law?"**
*   **Answer**: "Compute optimal training. For every doubling of model size, you should double training data. Ratio is roughly 20 tokens per parameter. (e.g., 70B model needs 1.4 Trillion tokens)."

**Q3: "How does DPO (Direct Preference Optimization) differ from PPO?"**
*   **Answer**: "PPO requires a separate Reward Model and Value Model (complex, unstable). DPO mathematically proves the Language Model *is* the Reward Model. We can optimize the policy directly on preference pairs $(y_w, y_l)$ using a simple binary cross-entropy loss."
