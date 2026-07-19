# The Complete AI Architecture Encyclopedia
## Model-by-Model, Algorithm-by-Algorithm Reference (2017–2026)

---

**Version:** 2.0 | **Date:** July 2026  
**Scope:** Every major architecture, every key algorithm, every frontier model

---

# TABLE OF CONTENTS

## PART I: FOUNDATIONAL ALGORITHMS
1. Tokenization Algorithms
2. Positional Encoding Algorithms
3. Normalization Algorithms
4. Attention Algorithms
5. Feed-Forward Algorithms
6. Mixture of Experts Algorithms
7. Training Algorithms
8. Inference Algorithms

## PART II: MODEL ARCHITECTURE BLUEPRINTS
9. GPT Family
10. BERT Family
11. T5 Family
12. PaLM Family
13. LLaMA Family
14. DeepSeek Family
15. Kimi Family (Moonshot AI)
16. GLM Family (Zhipu AI / z.ai)
17. Qwen Family
18. Mistral/Mixtral Family
19. Gemma/Gemini Family
20. Claude Family
21. State Space Models (Mamba)
22. Hybrid Architectures

## PART III: MATHEMATICAL APPENDIX
23. Complete Derivations
24. Complexity Analysis
25. Memory Layouts

---

# PART I: FOUNDATIONAL ALGORITHMS

---

## 1. TOKENIZATION ALGORITHMS

### 1.1 Byte-Pair Encoding (BPE)

**Origin:** Sennrich et al., 2016 (neural machine translation). Popularized by GPT-2.

**Algorithm:**
```
Input: Training corpus D, target vocabulary size V
Output: Vocabulary of subword units

1. Initialize vocabulary with all individual characters in D
2. Repeat until |vocabulary| = V:
   a. Tokenize D using current vocabulary
   b. Count all adjacent pairs of tokens
   c. Find the most frequent pair (A, B)
   d. Add merged token AB to vocabulary
   e. Replace all occurrences of (A, B) with AB in D
```

**Properties:**
- Greedy: always merges the most frequent pair
- Deterministic given vocabulary
- Handles OOV words by decomposing into subwords
- Vocabulary size typically 32K–160K

**Used by:** GPT-2, GPT-3, GPT-4, LLaMA, Codex, Claude (early)

### 1.2 Byte-level BPE (BBPE)

**Innovation:** Start with 256 byte values instead of characters. Can represent any Unicode text.

```
Initial vocabulary: {byte_0, byte_1, ..., byte_255}
Merge rules learned from byte sequences
```

**Advantage:** No OOV — any byte sequence can be tokenized.
**Used by:** GPT-2, RoBERTa, LLaMA

### 1.3 SentencePiece (Unigram Language Model)

**Origin:** Kudo & Richardson, 2018 (Google)

**Algorithm:**
```
1. Initialize large seed vocabulary (e.g., all substrings)
2. Repeat:
   a. Compute loss for each vocabulary item
   b. Remove items that least increase loss
   c. Until target vocabulary size reached
```

**Properties:**
- Language-agnostic (works on raw text)
- No pre-tokenization needed
- Supports BPE, unigram, and char/word modes
- Adds special tokens: <s>, </s>, <unk>, <pad>

**Used by:** T5, PaLM, mT5, XLNet

### 1.4 TikToken (BPE Optimized)

**Origin:** OpenAI, 2023

**Optimizations over standard BPE:**
- Regex-based pre-tokenization splits on natural boundaries
- Rust-based encoder (10× faster than Python)
- Special handling for code (whitespace, indentation)

**Regex pattern (cl100k_base):**
```
'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+'
```

**Used by:** GPT-4, GPT-4o, GPT-4-turbo

### 1.5 Tokenizer Comparison Table

| Model | Tokenizer | Vocab Size | Special Properties |
|-------|-----------|------------|-------------------|
| GPT-3 | BPE (50K) | 50,257 | End-of-text token |
| LLaMA 2 | SentencePiece | 32,000 | No <mask>, <pad> tokens |
| LLaMA 3 | TikToken-like | 128,256 | 3 special tokens for tool use |
| PaLM | SentencePiece | 256,000 | Multilingual, no <unk> |
| Kimi K2 | BPE variant | 160,000 | Extended for Chinese |
| DeepSeek-V3 | BPE | 128,000 | Includes code tokens |
| GLM-4.5 | SentencePiece | 100,000 | Bilingual (Chinese/English) |
| Mistral | SentencePiece | 32,000 | Same as LLaMA 2 |
| Gemma 2 | SentencePiece | 256,000 | Multilingual |

---

## 2. POSITIONAL ENCODING ALGORITHMS

### 2.1 Sinusoidal Positional Encoding

**Paper:** Attention Is All You Need (Vaswani et al., 2017)

**Formula:**

For position $pos$ and dimension $i$:

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

**Key Property (Linear Relationship):**

For any fixed offset $k$:

$$
PE_{pos+k} = f(PE_{pos}, PE_k)
$$

where $f$ is a linear transformation. This allows the model to learn relative positions.

**Pseudocode:**
```python
def sinusoidal_pe(max_len, d_model):
    pe = zeros(max_len, d_model)
    position = arange(0, max_len).unsqueeze(1)  # (max_len, 1)
    div_term = exp(arange(0, d_model, 2) * -(log(10000.0) / d_model))

    pe[:, 0::2] = sin(position * div_term)  # even dims
    pe[:, 1::2] = cos(position * div_term)  # odd dims
    return pe
```

**Limitations:**
- Fixed (not learned)
- Extrapolation beyond max_len degrades
- No explicit relative position modeling

### 2.2 Learned Absolute Positional Embeddings

**Paper:** BERT (Devlin et al., 2018), GPT-1 (Radford et al., 2018)

**Algorithm:**
```
Initialize: P ∈ R^(max_len × d_model)  (learned parameter)
Forward:    h = x_embed + P[:seq_len]
```

**Properties:**
- Simple, works well within training length
- Cannot extrapolate beyond max_len
- Each position has independent representation

### 2.3 Relative Position Representations

**Paper:** Shaw et al., 2018

**Idea:** Instead of adding position to input, modify attention scores:

$$
score_{ij} = q_i^T k_j + q_i^T r_{j-i}
$$

where $r_{j-i}$ is a learned embedding for relative distance $j-i$.

**Clipping:** Distances beyond $k$ are clipped: $r_{\min(j-i, k)}$

**Used by:** Transformer-XL, T5, some variants of BERT

### 2.4 ALiBi (Attention with Linear Biases)

**Paper:** Press et al., 2021

**Algorithm:**
```
For each head h:
    m_h = 2^(-8/h)  # head-specific slope
    bias_{ij} = -m_h * |i - j|  # linear distance penalty
    attention_logits += bias
```

**Key Insight:** No learned position parameters. The bias is a simple linear function of distance.

**Advantages:**
- No learned parameters for position
- Extrapolates naturally to longer sequences
- Stronger recency bias (closer tokens matter more)

**Disadvantages:**
- Can hurt performance on very long contexts where distant tokens matter
- Fixed penalty function, not adaptive

**Used by:** BLOOM, some MPT models, some Falcon variants

### 2.5 RoPE (Rotary Position Embedding)

**Paper:** RoFormer (Su et al., 2021)

**This is the most important positional encoding to understand.**

**Core Idea:** Rotate the query and key vectors by an angle proportional to their position. The dot product naturally encodes relative position.

**2D Case:**

For a 2D vector $x = (x_1, x_2)$ at position $m$:

$$
R_{\Theta,m} \cdot x = \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix}
$$

**General d-dimensional Case:**

For dimension pairs $(2i, 2i+1)$, use frequency $\theta_i = 10000^{-2i/d}$:

$$
R_{\Theta,m}^d = \begin{pmatrix}
\cos(m\theta_0) & -\sin(m\theta_0) & 0 & 0 & \cdots \\
\sin(m\theta_0) & \cos(m\theta_0) & 0 & 0 & \cdots \\
0 & 0 & \cos(m\theta_1) & -\sin(m\theta_1) & \cdots \\
0 & 0 & \sin(m\theta_1) & \cos(m\theta_1) & \cdots \\
\vdots & \vdots & \vdots & \vdots & \ddots
\end{pmatrix}
$$

**Application to Q and K:**

$$
\tilde{q}_m = R_{\Theta,m}^d \cdot q_m, \quad \tilde{k}_n = R_{\Theta,n}^d \cdot k_n
$$

**The Critical Property:**

$$
\tilde{q}_m^T \tilde{k}_n = q_m^T R_{\Theta,n-m}^d k_n
$$

The dot product depends **only on the relative distance** $n-m$!

**Pseudocode:**
```python
def apply_rope(x, seq_len, head_dim):
    # x: (batch, seq_len, n_heads, head_dim)
    # Split into pairs
    x1 = x[..., 0::2]  # even indices
    x2 = x[..., 1::2]  # odd indices

    # Compute frequencies
    freqs = 1.0 / (10000 ** (arange(0, head_dim, 2) / head_dim))
    angles = outer(arange(seq_len), freqs)  # (seq_len, head_dim/2)

    cos_angles = cos(angles)
    sin_angles = sin(angles)

    # Apply rotation
    rotated_x1 = x1 * cos_angles - x2 * sin_angles
    rotated_x2 = x1 * sin_angles + x2 * cos_angles

    # Interleave back
    return stack([rotated_x1, rotated_x2], dim=-1).flatten(-2)
```

**Used by:** LLaMA, PaLM, GPT-NeoX, Mistral, DeepSeek, Kimi, GLM, Qwen, Gemma, Claude (reportedly)

### 2.6 YaRN (Yet another RoPE extension method)

**Paper:** Peng et al., 2023

**Problem:** RoPE trained on length $L$ degrades at length $L' > L$.

**Solution:** Interpolate frequencies + temperature scaling

```
1. Compute scale factor: s = L' / L
2. Interpolate frequencies: θ'_i = θ_i / s
3. Apply temperature scaling to attention logits
4. Optionally scale attention by 1/log(s)
```

**Variants:**
- **NTK-aware:** Non-linear interpolation (better for code)
- **Dynamic NTK:** Compute scale at runtime
- **YaRN:** Combines interpolation + magnitude correction

**Extension factors achieved:** 2×–8× original context length

**Used by:** LLaMA 2 extended, many fine-tuned models

### 2.7 3D-RoPE (for Vision-Language Models)

**Origin:** GLM-4.5V, some multimodal models

**Idea:** Extend RoPE to 3D spatial coordinates $(x, y, t)$:

$$
R_{3D}(x, y, t) = R_{\theta_x}(x) \cdot R_{\theta_y}(y) \cdot R_{\theta_t}(t)
$$

**Used for:** Spatial understanding in vision-language models

---

## 3. NORMALIZATION ALGORITHMS

### 3.1 Batch Normalization (for reference)

**Not used in Transformers** (sequence length varies), but historically important:

$$
BN(x) = \gamma \cdot \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} + \beta
$$

where $\mu_B, \sigma_B$ are computed over the batch dimension.

### 3.2 Layer Normalization

**Paper:** Ba et al., 2016

**Formula:**

For input $x \in \mathbb{R}^d$:

$$
\mu = \frac{1}{d}\sum_{i=1}^{d} x_i, \quad \sigma^2 = \frac{1}{d}\sum_{i=1}^{d} (x_i - \mu)^2
$$

$$
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

$$
LN(x) = \gamma \odot \hat{x} + \beta
$$

**Pseudocode:**
```python
def layer_norm(x, gamma, beta, eps=1e-6):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_norm = (x - mean) / sqrt(var + eps)
    return gamma * x_norm + beta
```

**Properties:**
- Normalizes across feature dimension (not batch)
- Learns scale ($\gamma$) and shift ($\beta$) parameters
- 2d parameters total

### 3.3 RMSNorm (Root Mean Square Layer Normalization)

**Paper:** Zhang & Sennrich, 2019

**Formula:**

$$
RMS(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}
$$

$$
RMSNorm(x) = \frac{x}{RMS(x)} \odot \gamma
$$

**Pseudocode:**
```python
def rms_norm(x, gamma, eps=1e-6):
    rms = sqrt(mean(x ** 2, dim=-1, keepdim=True) + eps)
    return x / rms * gamma
```

**Comparison with LayerNorm:**

| Aspect | LayerNorm | RMSNorm |
|--------|-----------|---------|
| Mean subtraction | Yes | No |
| Variance computation | Yes (full variance) | No (RMS only) |
| Learned shift ($\beta$) | Yes | No |
| Parameters | 2d | d |
| FLOPs | More | Fewer |
| Memory traffic | More | Fewer |
| Empirical performance | Equivalent | Equivalent or better |

**Why RMSNorm is faster:**
- Skips mean subtraction (1 fewer op)
- No $\beta$ parameter (fewer memory loads)
- In memory-bound regimes, parameter loading dominates
- RMSNorm is ~25% of runtime in some workloads despite <1% of FLOPs

**Used by:** LLaMA, PaLM, Mistral, DeepSeek, Kimi, GLM, Qwen, Gemma

### 3.4 Pre-Norm vs Post-Norm

**Post-Norm (Original Transformer, 2017):**
```
x → Attention → Add → LayerNorm → FFN → Add → LayerNorm
```

**Pre-Norm (Modern Standard):**
```
x → LayerNorm → Attention → Add → LayerNorm → FFN → Add
```

**Why Pre-Norm is better:**

Gradient flow analysis:
- Post-Norm: gradients must pass through LayerNorm before residual
- Pre-Norm: gradients flow directly through residual connection

For a network with $L$ layers:
- Post-Norm: effective gradient path length ≈ $L$
- Pre-Norm: effective gradient path length ≈ 1 (direct)

**This is why Pre-Norm enables training 100+ layer networks.**

**DeepNorm (Wang et al., 2022):**

For very deep networks, scale the residual:

$$
x_{l+1} = LN(\alpha \cdot x_l + G_l(x_l))
$$

where $\alpha = (2L)^{1/4}$ for encoder, $(3L)^{1/4}$ for decoder.

**Used by:** Some very deep models (e.g., GLM-130B)

### 3.5 QK-Normalization

**Paper:** Various (Gemma 2, DCLM, OLMo 2)

**Algorithm:**
```python
def qk_norm_attention(Q, K, V):
    Q = layer_norm(Q)  # or RMSNorm
    K = layer_norm(K)
    scores = Q @ K.T / sqrt(d_k)
    attn = softmax(scores)
    return attn @ V
```

**Why:** Prevents dot products from growing too large, which makes softmax saturate (all probability mass on one token → gradient vanishing).

### 3.6 QK-Clipping (MuonClip)

**Origin:** Moonshot AI, Kimi K2

**Problem:** Muon optimizer (more efficient than AdamW) causes training instability in attention layers.

**Solution:** Explicitly clip the magnitude of Q and K weight matrices:

```python
def qk_clip(W_q, W_k, max_norm=1.0):
    W_q = W_q / max(1, norm(W_q) / max_norm)
    W_k = W_k / max(1, norm(W_k) / max_norm)
    return W_q, W_k
```

**Result:** Enabled training Kimi K2 (1T params) with **zero loss spikes**.

---

## 4. ATTENTION ALGORITHMS

### 4.1 Standard Multi-Head Attention (MHA)

**Complete Algorithm:**

```python
def multi_head_attention(X, W_q, W_k, W_v, W_o, n_heads, mask=None):
    batch, seq_len, d_model = X.shape
    d_k = d_model // n_heads

    # Linear projections
    Q = X @ W_q    # (batch, seq, d_model)
    K = X @ W_k
    V = X @ W_v

    # Reshape for multi-head
    Q = Q.view(batch, seq_len, n_heads, d_k).transpose(1, 2)  # (batch, heads, seq, d_k)
    K = K.view(batch, seq_len, n_heads, d_k).transpose(1, 2)
    V = V.view(batch, seq_len, n_heads, d_k).transpose(1, 2)

    # Scaled dot-product attention
    scores = Q @ K.transpose(-2, -1) / sqrt(d_k)  # (batch, heads, seq, seq)

    # Apply causal mask for decoder
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attn_weights = softmax(scores, dim=-1)
    out = attn_weights @ V  # (batch, heads, seq, d_k)

    # Concatenate heads
    out = out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)

    # Output projection
    return out @ W_o
```

**Complexity:**
- Time: $O(n^2 \cdot d)$
- Memory: $O(n^2)$ for attention matrix + $O(n \cdot d)$ for Q, K, V

### 4.2 Multi-Query Attention (MQA)

**Paper:** Shazeer, 2019

**Modification:** All query heads share the same K and V.

```python
def multi_query_attention(X, W_q, W_k_shared, W_v_shared, W_o, n_heads):
    # Q: (batch, seq, n_heads * d_k)
    # K, V: (batch, seq, d_k)  -- SHARED across heads

    Q = X @ W_q
    K = X @ W_k_shared  # Only ONE K projection
    V = X @ W_v_shared  # Only ONE V projection

    Q = Q.view(batch, seq, n_heads, d_k).transpose(1, 2)
    K = K.unsqueeze(1)  # (batch, 1, seq, d_k) -- broadcast to all heads
    V = V.unsqueeze(1)

    scores = Q @ K.transpose(-2, -1) / sqrt(d_k)
    attn = softmax(scores, dim=-1)
    out = attn @ V

    return out.transpose(1, 2).reshape(batch, seq, d_model) @ W_o
```

**KV Cache Size:** Reduced by factor of $n_{heads}$.

**Quality Impact:** Slight degradation because K/V have less representational capacity.

**Used by:** PaLM, some inference-optimized variants

### 4.3 Grouped Query Attention (GQA)

**Paper:** Ainslie et al., 2023

**Modification:** Query heads are divided into groups. Each group shares K/V.

```python
def grouped_query_attention(X, W_q, W_k, W_v, W_o, n_q_heads, n_kv_heads):
    # n_q_heads = 32, n_kv_heads = 8
    # Each KV head is shared by 4 query heads

    Q = X @ W_q  # (batch, seq, n_q_heads * d_k)
    K = X @ W_k  # (batch, seq, n_kv_heads * d_k)
    V = X @ W_v

    Q = Q.view(batch, seq, n_q_heads, d_k).transpose(1, 2)
    K = K.view(batch, seq, n_kv_heads, d_k).transpose(1, 2)
    V = V.view(batch, seq, n_kv_heads, d_k).transpose(1, 2)

    # Repeat K, V to match query heads
    K = K.repeat_interleave(n_q_heads // n_kv_heads, dim=1)
    V = V.repeat_interleave(n_q_heads // n_kv_heads, dim=1)

    scores = Q @ K.transpose(-2, -1) / sqrt(d_k)
    attn = softmax(scores, dim=-1)
    out = attn @ V

    return out.transpose(1, 2).reshape(batch, seq, d_model) @ W_o
```

**KV Cache Size:** Reduced by factor of $n_{q\_heads} / n_{kv\_heads}$.

**Used by:** LLaMA 2/3, Mistral, Gemma, Qwen, most modern models

### 4.4 Multi-Head Latent Attention (MLA)

**Paper:** DeepSeek-V2 (DeepSeek-AI, 2024)

**This is the most important attention innovation since RoPE.**

**Core Idea:** Compress K and V into a single low-rank latent vector during inference. Expand back during attention computation.

**Architecture:**

```
During training:
  h → W_DQ (down-projection for Q) → c_Q
  h → W_DKV (down-projection for KV) → c_KV

During inference (KV cache stores only c_KV):
  c_KV → W_UK (up-projection) → K
  c_KV → W_UV (up-projection) → V

  Q = c_Q @ W_UQ
  K = c_KV @ W_UK
  V = c_KV @ W_UV

  Attention(Q, K, V) as usual
```

**Pseudocode:**
```python
def mla_attention(X, W_dq, W_dkv, W_uq, W_uk, W_uv, W_o, n_heads, d_c):
    # d_c: compressed dimension (much smaller than d_model)

    # Compress
    c_q = X @ W_dq      # (batch, seq, d_c)
    c_kv = X @ W_dkv    # (batch, seq, d_c) -- THIS IS CACHED

    # Expand
    Q = c_q @ W_uq      # (batch, seq, n_heads * d_k)
    K = c_kv @ W_uk
    V = c_kv @ W_uv

    # Standard multi-head attention
    Q = Q.view(batch, seq, n_heads, d_k).transpose(1, 2)
    K = K.view(batch, seq, n_heads, d_k).transpose(1, 2)
    V = V.view(batch, seq, n_heads, d_k).transpose(1, 2)

    scores = Q @ K.transpose(-2, -1) / sqrt(d_k)
    attn = softmax(scores, dim=-1)
    out = attn @ V

    return out.transpose(1, 2).reshape(batch, seq, d_model) @ W_o
```

**KV Cache Comparison:**

| Method | Per-token Cache Size | Relative to MHA |
|--------|---------------------|-----------------|
| MHA | $2 \cdot L \cdot H \cdot d_h$ | 1× |
| MQA | $2 \cdot L \cdot d_h$ | $1/H$ × |
| GQA (G=8, H=128) | $2 \cdot L \cdot 8 \cdot d_h$ | $1/16$ × |
| MLA (DeepSeek-V2) | $L \cdot d_c$ | ~$1/7$ × |

Where $L$ = layers, $H$ = heads, $d_h$ = head dim, $d_c$ = compressed dim.

**Used by:** DeepSeek-V2/V3, Kimi K2/K3

### 4.5 FlashAttention

**Paper:** Dao et al., 2022 (Stanford)

**Problem:** Standard attention materializes the full $N \times N$ attention matrix in GPU HBM. For $N=4096$, this is ~64MB in FP16. For $N=128K$, it's ~32GB.

**Key Insight:** The attention matrix doesn't need to be stored. We can compute softmax incrementally.

**Online Softmax:**

Standard softmax:
```
m = max(x)
y_i = exp(x_i - m) / sum(exp(x_j - m))
```

Online softmax (process in chunks):
```
m_old = -inf
l_old = 0
for chunk in chunks:
    m_new = max(m_old, max(chunk))
    l_new = exp(m_old - m_new) * l_old + sum(exp(chunk - m_new))
    m_old = m_new
    l_old = l_new
```

**FlashAttention Algorithm (Tiling):**

```python
def flash_attention(Q, K, V, block_size):
    N, d = Q.shape
    O = zeros(N, d)      # Output
    L = zeros(N)         # Running log-sum-exp
    m = full(N, -inf)    # Running max

    for i in range(0, N, block_size):      # Tile over queries
        Qi = Q[i:i+block_size]
        Oi = zeros(block_size, d)
        mi = full(block_size, -inf)
        li = zeros(block_size)

        for j in range(0, N, block_size):  # Tile over keys
            Kj = K[j:j+block_size]
            Vj = V[j:j+block_size]

            Sij = Qi @ Kj.T                  # (block, block) scores
            mij = rowmax(Sij)
            lij = rowsum(exp(Sij - mij))

            # Update running statistics
            mi_new = max(mi, mij)
            li_new = exp(mi - mi_new) * li + exp(mij - mi_new) * lij

            # Update output
            Oi = (exp(mi - mi_new) * li * Oi + exp(mij - mi_new) * (exp(Sij - mij) @ Vj)) / li_new

            mi = mi_new
            li = li_new

        O[i:i+block_size] = Oi

    return O
```

**Memory Complexity:** $O(N)$ instead of $O(N^2)$
**Speedup:** 2-4× (limited by memory bandwidth, not compute)

**FlashAttention-2 improvements:**
- Better work partitioning across warps
- Reduced non-matmul FLOPs
- ~2× faster than v1

**FlashAttention-3 improvements:**
- FP8 support
- Asynchronous operations (overlap compute and memory)
- ~1.5× faster than v2 on H100

### 4.6 MoBA: Mixture of Block Attention

**Paper:** Moonshot AI, 2025

**Core Idea:** Apply MoE gating to attention blocks. The context is divided into blocks; each query learns which blocks to attend to.

**Algorithm:**
```python
def moba_attention(Q, K, V, block_size, top_k):
    N, d = Q.shape
    n_blocks = ceil(N / block_size)

    # Divide K, V into blocks
    K_blocks = [K[i*block_size:(i+1)*block_size] for i in range(n_blocks)]
    V_blocks = [V[i*block_size:(i+1)*block_size] for i in range(n_blocks)]

    # Compute routing scores (which blocks each query should attend to)
    # Using a lightweight gating network
    gate_scores = gate_network(Q)  # (N, n_blocks)
    topk_indices = topk(gate_scores, k=top_k)  # (N, top_k)

    O = zeros(N, d)
    for i in range(N):
        qi = Q[i:i+1]  # (1, d)
        blocks_to_use = topk_indices[i]

        # Gather relevant K, V blocks
        K_selected = concat([K_blocks[j] for j in blocks_to_use])
        V_selected = concat([V_blocks[j] for j in blocks_to_use])

        # Standard attention on selected blocks
        scores = qi @ K_selected.T / sqrt(d)
        attn = softmax(scores, dim=-1)
        O[i] = attn @ V_selected

    return O
```

**Performance:**
- 6.5× speedup at 1M context
- Scales to 10M tokens
- Can switch between full and sparse attention seamlessly

**Used by:** Kimi (production deployment)

### 4.7 Kimi Delta Attention (KDA)

**Origin:** Moonshot AI, 2025

**Type:** Linear attention variant

**Standard attention:** $O = \text{softmax}(QK^T)V$

**Linear attention:** Replace softmax with a feature map $\phi$:

$$
O = \frac{\phi(Q)\phi(K)^T V}{\phi(Q)\phi(K)^T \mathbf{1}}
$$

Using associativity:

$$
O = \frac{\phi(Q)(\phi(K)^T V)}{\phi(Q)(\phi(K)^T \mathbf{1})}
$$

Now the computation is $O(N \cdot d^2)$ instead of $O(N^2 \cdot d)$!

**KDA specifics:**
- Custom feature map $\phi$ designed for stability
- Maintains quality close to full attention
- Dedicated FlashKDA CUDA kernels for efficient inference

**Used by:** Kimi Linear (48B total, 3B active)

---

## 5. FEED-FORWARD ALGORITHMS

### 5.1 Standard FFN (ReLU)

```
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
```

- $W_1 \in \mathbb{R}^{d \times d_{ff}}$, $W_2 \in \mathbb{R}^{d_{ff} \times d}$
- Typically $d_{ff} = 4d$
- Parameters: $2 \cdot d \cdot d_{ff} = 8d^2$

### 5.2 GELU FFN

```
GELU(x) = x · Φ(x) = x · 0.5 · (1 + tanh[√(2/π) · (x + 0.044715 · x³)])
```

Smoother than ReLU. Used in BERT, GPT-3.

### 5.3 SwiGLU

**Paper:** Shazeer, 2020 ("GLU Variants Improve Transformer")

**Formula:**

$$
\text{SwiGLU}(x) = (\text{Swish}_1(xW) \odot xV)W_2
$$

where:
- $\text{Swish}_1(x) = x \cdot \sigma(x)$ (sigmoid-weighted linear unit)
- $\odot$ is element-wise multiplication
- $W, V \in \mathbb{R}^{d \times d_{ff}}$, $W_2 \in \mathbb{R}^{d_{ff} \times d}$

**Pseudocode:**
```python
def swiglu(x, W, V, W2):
    a = x @ W
    b = x @ V
    swish = a * sigmoid(a)
    return (swish * b) @ W2
```

**Why 3 matrices?** The gating mechanism (Swish + element-wise multiply) requires two input projections.

**Parameter count:** $3 \cdot d \cdot d_{ff}$

To keep parameter count similar to standard FFN, use $d_{ff} = \frac{2}{3} \cdot 4d = \frac{8}{3}d$.

**Used by:** LLaMA, PaLM, Mistral, DeepSeek, Kimi, GLM, Qwen, Gemma, Claude

### 5.4 GEGLU

Same as SwiGLU but with GELU instead of Swish:

$$
\text{GEGLU}(x) = (\text{GELU}(xW) \odot xV)W_2
$$

**Used by:** PaLM (some variants)

---

## 6. MIXTURE OF EXPERTS ALGORITHMS

### 6.1 Basic MoE Layer

```python
def moe_layer(x, experts, router, top_k):
    # x: (batch, seq, d_model)
    # experts: list of N expert FFNs
    # router: linear layer (d_model -> N)

    # Compute routing scores
    logits = router(x)  # (batch, seq, N)

    # Select top-k experts
    weights, indices = torch.topk(softmax(logits, dim=-1), k=top_k)

    # Normalize weights
    weights = weights / weights.sum(dim=-1, keepdim=True)

    # Compute expert outputs
    output = zeros_like(x)
    for i in range(top_k):
        expert_idx = indices[..., i]
        expert_weight = weights[..., i:i+1]

        # Route each token to its selected expert
        for b in range(batch):
            for s in range(seq):
                e = expert_idx[b, s]
                output[b, s] += expert_weight[b, s] * experts[e](x[b, s])

    return output
```

### 6.2 Load Balancing

**Problem:** Router may send all tokens to the same few experts.

**Auxiliary Loss (Switch Transformer):**

$$
\mathcal{L}_{\text{aux}} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i
$$

where:
- $f_i = \frac{1}{T} \sum_{t=1}^{T} \mathbb{1}[\text{expert}_i \text{ selected for token } t]$ (fraction of tokens to expert $i$)
- $P_i = \frac{1}{T} \sum_{t=1}^{T} p_i(t)$ (average routing probability for expert $i$)

**Aux-Loss-Free (DeepSeek-V3):**

Add bias terms $b_i$ to routing scores:

```
routing_score_i = softmax(logit_i + b_i)
```

- $b_i$ is updated based on expert load (not backpropagated)
- If expert $i$ is overloaded, decrease $b_i$
- If expert $i$ is underloaded, increase $b_i$

**Result:** No auxiliary loss needed, better gradient quality.

### 6.3 DeepSeekMoE Design

**Fine-Grained Expert Segmentation:**
- Instead of 8 large experts, use 256 small experts
- Activate more experts (e.g., 8 out of 256 vs 1 out of 8)
- Finer specialization

**Shared Expert Isolation:**
- 1-2 experts are ALWAYS activated
- Capture "common knowledge" (basic language, general reasoning)
- Routed experts specialize in niche domains

**Pseudocode:**
```python
def deepseek_moe(x, shared_experts, routed_experts, router, top_k):
    # Shared experts (always active)
    shared_output = sum([expert(x) for expert in shared_experts])

    # Routed experts (selective)
    logits = router(x)
    weights, indices = topk(softmax(logits), k=top_k)

    routed_output = zeros_like(x)
    for i in range(top_k):
        e = routed_experts[indices[..., i]]
        routed_output += weights[..., i:i+1] * e(x)

    return shared_output + routed_output
```

### 6.4 Sigmoid Gates (GLM-4.5/5.2)

Instead of softmax over all experts, use independent sigmoid:

```python
# Softmax (standard)
weights = softmax(logits)  # Sum to 1, competitive

# Sigmoid (GLM)
weights = sigmoid(logits)  # Independent, each 0-1
weights = topk(weights, k) / sum(topk(weights, k))  # Renormalize
```

**Advantage:** More sample-efficient. Each expert's gate is learned independently.

---

## 7. TRAINING ALGORITHMS

### 7.1 AdamW Optimizer

**Standard for LLM training.**

```python
def adamw(params, grads, lr, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
    for p, g in zip(params, grads):
        m = beta1 * m + (1 - beta1) * g      # First moment
        v = beta2 * v + (1 - beta2) * g**2   # Second moment

        m_hat = m / (1 - beta1**t)           # Bias correction
        v_hat = v / (1 - beta2**t)

        # Decoupled weight decay
        p = p - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * p)
```

**Hyperparameters:**
- $\beta_1 = 0.9$ (momentum)
- $\beta_2 = 0.95$ (for LLMs, slightly lower than default 0.999)
- Weight decay: 0.1 (LLaMA), 0.01 (some models)
- Learning rate: ~1e-4 to 3e-4 for large models

### 7.2 Muon Optimizer

**Paper:** Moonshot AI, 2025

**Core Idea:** Use Newton-Schulz iterations to orthogonalize gradients.

**Algorithm:**
```python
def muon_step(grad, lr, momentum=0.9, n_iter=5):
    # 1. Compute momentum
    m = momentum * m + grad

    # 2. Newton-Schulz orthogonalization
    G = m
    for _ in range(n_iter):
        G = 1.5 * G - 0.5 * G @ G.T @ G  # Orthogonalize

    # 3. Update
    param -= lr * G
```

**Advantages:**
- ~2× more sample-efficient than AdamW
- Better gradient conditioning
- Used by Moonshot to train Kimi K2 with zero loss spikes

**Disadvantages:**
- Only works well for 2D matrices (not embeddings, LayerNorm)
- Requires QK-Clip for stability in attention layers

### 7.3 Learning Rate Scheduling

**Cosine with Warmup:**

```python
def lr_schedule(step, warmup_steps, max_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    else:
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        return min_lr + (max_lr - min_lr) * 0.5 * (1 + cos(pi * progress))
```

**Typical values:**
- Warmup: 2,000–10,000 steps
- Max LR: 1e-4 to 3e-4
- Min LR: 1e-5 to max_lr/10

### 7.4 Loss Functions

**Next-Token Prediction (Standard):**

$$
\mathcal{L}_{\text{NTP}} = -\sum_{t=1}^{T} \log P(x_t | x_{<t})
$$

**Multi-Token Prediction (MTP):**

$$
\mathcal{L}_{\text{MTP}} = -\sum_{t=1}^{T} \sum_{k=1}^{K} \log P(x_{t+k} | x_{\leq t})
$$

where $K$ is the number of future tokens to predict (typically 1-4).

**Z-Loss (Output Stabilizer):**

$$
\mathcal{L}_{\text{Z}} = \alpha \cdot \log^2(Z)
$$

where $Z = \sum_{i} \exp(z_i)$ is the softmax normalizer.

**Purpose:** Prevents logits from growing too large, which causes numerical instability.

**Used by:** PaLM, Chameleon, some multimodal models

### 7.5 Training Stability Techniques

| Technique | What it does | Used by |
|-----------|-------------|---------|
| **Gradient Clipping** | Clip gradient norm to max value | Universal |
| **Loss Spike Recovery** | Roll back to last checkpoint on spike | DeepSeek |
| **QK-Norm** | Normalize Q, K before attention | Gemma 2, DCLM |
| **QK-Clip** | Clip Q, K weight magnitudes | Kimi K2 |
| **Soft Capping** | tanh(logits / cap) * cap | Gemma, some NVIDIA |
| **FP8 Training** | Train in 8-bit floating point | DeepSeek-V3 |
| **Tensor Parallelism** | Split layers across GPUs | All large models |
| **Pipeline Parallelism** | Split model depth across GPUs | All large models |
| **Expert Parallelism** | Distribute experts across GPUs | MoE models |

---

## 8. INFERENCE ALGORITHMS

### 8.1 KV Cache

**Algorithm:**
```python
def generate_with_kv_cache(model, prompt, max_new_tokens):
    # 1. Process prompt (prefill)
    k_cache, v_cache = [], []
    x = embed(prompt)
    for layer in model.layers:
        q, k, v = layer.attention.project(x)
        k_cache.append(k)
        v_cache.append(v)
        x = layer(x)  # Full forward pass

    # 2. Generate tokens one at a time
    for _ in range(max_new_tokens):
        # Only compute K, V for the NEW token
        new_k = layer.attention.project_k(x[:, -1:])
        new_v = layer.attention.project_v(x[:, -1:])

        # Append to cache
        k_cache[layer] = concat(k_cache[layer], new_k, dim=1)
        v_cache[layer] = concat(v_cache[layer], new_v, dim=1)

        # Attention uses full cache
        attn_out = attention(q, k_cache[layer], v_cache[layer])

        # Predict next token
        logits = model.lm_head(attn_out)
        next_token = sample(logits)

        x = concat(x, embed(next_token), dim=1)

    return x
```

**Memory per token (FP16):**

$$
M_{\text{cache}} = 2 \cdot L \cdot H_{\text{kv}} \cdot d_h \cdot 2 \text{ bytes}
$$

For a 32-layer, 8-head GQA model with 128-dim heads:
- $M = 2 \cdot 32 \cdot 8 \cdot 128 \cdot 2 = 131,072$ bytes = **128 KB per token**

At 128K context: **16 GB** just for KV cache!

### 8.2 Speculative Decoding

**Paper:** Leviathan et al., 2022

**Idea:** Use a small "draft" model to predict multiple tokens, then verify with the large model in parallel.

```python
def speculative_decode(draft_model, target_model, prompt, K=5):
    # 1. Draft model generates K tokens
    draft_tokens = draft_model.generate(prompt, max_new=K)

    # 2. Target model verifies all K+1 tokens in parallel
    all_tokens = concat([prompt[-1]], draft_tokens)
    logits = target_model(all_tokens)  # Single forward pass!

    # 3. Accept tokens until disagreement
    accepted = []
    for i in range(K):
        if sample(logits[i]) == draft_tokens[i]:
            accepted.append(draft_tokens[i])
        else:
            accepted.append(sample(logits[i]))
            break

    return accepted
```

**Speedup:** 2-3× if draft model is good and cheap.

**Variants:**
- **Medusa:** Add multiple heads to the same model (no separate draft model)
- **EAGLE:** Train a small autoencoder as draft model
- **MTP:** Built-in multi-token prediction (DeepSeek-V3, GLM-5.2)

### 8.3 Quantization

**Methods:**

| Method | Bits | Accuracy Loss | Speedup |
|--------|------|--------------|---------|
| FP16 | 16 | None | Baseline |
| INT8 | 8 | <1% | 2× |
| INT4 (GPTQ) | 4 | 2-5% | 4× |
| FP8 (NVIDIA) | 8 | <1% | 2× |
| AWQ | 4 | 1-3% | 4× |
| GGUF | 4-8 | 2-5% | 2-4× |

**GPTQ Algorithm (simplified):**
```python
def gptq_quantize(W, bits=4):
    # 1. Compute Hessian H = X^T X
    # 2. For each column w_i of W:
    #    a. Quantize: w_q = round(w_i / scale) * scale
    #    b. Compute error: err = w_i - w_q
    #    c. Update remaining weights: w_{i+1:} -= err * H_{i,i+1:} / H_{i,i}
    # 3. Return quantized weights
```

---

# PART II: MODEL ARCHITECTURE BLUEPRINTS

---

## 9. GPT FAMILY

### 9.1 GPT-1 (2018)

| Spec | Value |
|------|-------|
| Parameters | 117M |
| Layers | 12 |
| d_model | 768 |
| Heads | 12 |
| d_ff | 3,072 |
| Context | 512 |
| Vocabulary | 40,000 (BPE) |
| Position | Learned absolute |
| Norm | Post-LayerNorm |
| FFN | ReLU |
| Attention | MHA |
| Training tokens | ~1B (BooksCorpus) |

**Architecture:**
```
Input → Token Embed + Pos Embed → [Decoder Block]×12 → LayerNorm → LM Head
```

**Innovation:** Unsupervised pre-training + supervised fine-tuning paradigm.

### 9.2 GPT-2 (2019)

| Spec | Small | Medium | Large | XL |
|------|-------|--------|-------|-----|
| Params | 124M | 355M | 774M | 1.5B |
| Layers | 12 | 24 | 36 | 48 |
| d_model | 768 | 1,024 | 1,280 | 1,600 |
| Heads | 12 | 16 | 20 | 25 |
| Context | 1,024 | 1,024 | 1,024 | 1,024 |

**Key changes from GPT-1:**
- LayerNorm moved to input of sublayers (pre-norm)
- Final LayerNorm added after last block
- Weight initialization scaled by $1/\sqrt{N}$ where $N$ is residual layer count
- Byte-level BPE (no OOV)

### 9.3 GPT-3 (2020)

| Spec | Value |
|------|-------|
| Parameters | 175B |
| Layers | 96 |
| d_model | 12,288 |
| Heads | 96 |
| d_head | 128 |
| d_ff | 49,152 |
| Context | 2,048 |
| Vocabulary | 50,257 (BPE) |
| Position | Learned absolute |
| Norm | Pre-LayerNorm |
| FFN | GeLU |
| Attention | MHA |
| Training tokens | 300B |
| Batch size | 3.2M tokens |
| Learning rate | 0.6×10⁻⁴ (warmup), 0.06×10⁻⁴ (cosine decay) |

**Architecture diagram:**
```
Input (2,048 tokens)
  ↓
Token Embedding (50,257 × 12,288)
  ↓
[Pre-LN → MHA (96 heads, 128 dim) → Residual]×96
  ↓
[Pre-LN → GeLU FFN (12,288 → 49,152 → 12,288) → Residual]×96
  ↓
Final LayerNorm
  ↓
Output Projection (12,288 × 50,257)
  ↓
Softmax → Next token probability
```

**Training details:**
- Optimizer: Adam (β₁=0.9, β₂=0.95, ε=10⁻⁸)
- Gradient clipping: 1.0
- Weight decay: 0.1
- Dropout: 0.1 (attention + residual)
- Precision: FP16 with loss scaling

**Emergent capabilities:**
- Few-shot learning (no gradient updates)
- In-context learning
- Arithmetic, translation, question answering

### 9.4 GPT-4 (2023)

**Closed model — limited public details.**

| Known/Estimated Spec | Value |
|---------------------|-------|
| Total parameters | ~1.8T (estimated) |
| Active parameters | ~200B (estimated, MoE) |
| Context | 8K, 32K variants |
| Architecture | Decoder-only MoE |
| Training data | Web pages, books, code, images |
| RLHF | Yes (extensive) |
| Multimodal | Yes (text + images) |

**Key innovations (inferred):**
- MoE architecture (8 experts, 2 active)
- Advanced RLHF with constitutional AI
- Multimodal training (interleaved text and images)
- Tool use training

---

## 10. BERT FAMILY

### 10.1 BERT (2018)

| Spec | Base | Large |
|------|------|-------|
| Parameters | 110M | 340M |
| Layers | 12 | 24 |
| d_model | 768 | 1,024 |
| Heads | 12 | 16 |
| d_ff | 3,072 | 4,096 |
| Context | 512 | 512 |

**Architecture:** Encoder-only

**Training objectives:**
1. **Masked Language Modeling (MLM):** Randomly mask 15% of tokens, predict them
2. **Next Sentence Prediction (NSP):** Predict if sentence B follows sentence A

**MLM masking strategy:**
- 80% of time: replace with [MASK]
- 10% of time: replace with random token
- 10% of time: keep original

### 10.2 RoBERTa (2019)

**Optimizations over BERT:**
- Train longer (160GB text vs 16GB)
- Larger batches (8K vs 256)
- Remove NSP (not helpful)
- Dynamic masking (different mask per epoch)
- Full sentences (not pairs)
- Byte-level BPE (50K vocab)

**Result:** Matches BERT-Large with BERT-Base parameters.

---

## 11. T5 FAMILY

### 11.1 T5 (2019)

| Spec | Value |
|------|-------|
| Parameters | 11B (largest variant) |
| Architecture | Encoder-Decoder |
| Layers | 24 (encoder) + 24 (decoder) |
| d_model | 1,024 |
| Heads | 16 |
| d_ff | 65,536 (GLU variant) |
| Context | 512 |
| Position | Relative position bias |

**Unified text-to-text framework:**
- All NLP tasks cast as text generation
- Prefix tasks with special tokens: "translate English to German: ..."

**Relative position bias:**
```
Attention score += b_{j-i}
```
where $b_{j-i}$ is a learned scalar for each relative distance.

### 11.2 UL2 (2022)

**Mixture of Denoisers:**
- R-denoiser: causal language modeling (prefix)
- S-denoiser: span corruption (like T5)
- X-denoiser: extreme span corruption

**Result:** Single model that handles both causal and bidirectional tasks.

---

## 12. PaLM FAMILY

### 12.1 PaLM (2022)

| Spec | Value |
|------|-------|
| Parameters | 540B |
| Layers | 118 |
| d_model | 18,432 |
| Heads | 48 |
| d_head | 256 (very large!) |
| d_ff | 73,728 |
| Context | 2,048 |
| Vocabulary | 256,000 (SentencePiece) |
| Position | RoPE |
| Norm | Pre-RMSNorm |
| FFN | GeGLU |
| Attention | MQA |
| Parallel layers | Yes (attention and FFN in parallel) |
| Training tokens | 780B |

**Parallel layers (unique to PaLM):**
```
Standard:  x → Attention → Add → FFN → Add
PaLM:      x ──► Attention ──┐
           x ──► FFN ────────├──► Add → output
```

**Training system:** Pathways (Google's distributed ML system)
- 6,144 TPU v4 chips
- 2 pods of 3,072 chips each
- Data + model + pipeline parallelism

### 12.2 PaLM 2 (2023)

| Spec | Value |
|------|-------|
| Parameters | Not disclosed (smaller than PaLM but better) |
| Architecture | Improved PaLM |
| Context | 8K, 32K variants |
| Multilingual | Yes (100+ languages) |
| RLHF | Yes |

**Key improvements:**
- Better data curation
- More compute-efficient architecture
- Multilingual from the start

---

## 13. LLaMA FAMILY

### 13.1 LLaMA (2023)

| Spec | 7B | 13B | 33B | 65B |
|------|-----|-----|-----|-----|
| Layers | 32 | 40 | 60 | 80 |
| d_model | 4,096 | 5,120 | 6,656 | 8,192 |
| Heads | 32 | 40 | 52 | 64 |
| d_head | 128 | 128 | 128 | 128 |
| d_ff | 11,008 | 13,824 | 17,920 | 22,016 |
| Context | 2,048 | 2,048 | 2,048 | 2,048 |
| Vocabulary | 32,000 | 32,000 | 32,000 | 32,000 |
| Position | RoPE | RoPE | RoPE | RoPE |
| Norm | Pre-RMSNorm | Pre-RMSNorm | Pre-RMSNorm | Pre-RMSNorm |
| FFN | SwiGLU | SwiGLU | SwiGLU | SwiGLU |
| Attention | MHA | MHA | MHA | MHA |
| Training tokens | 1.0T | 1.0T | 1.4T | 1.4T |

**Architecture innovations:**
- Pre-RMSNorm (no learned bias)
- SwiGLU activation
- RoPE positional encoding
- No bias in linear layers
- Weight tying (input/output embeddings)

**Training details:**
- Optimizer: AdamW (β₁=0.9, β₂=0.95)
- LR: 3e-4 (7B), 1.5e-4 (13B), 1e-4 (33B), 8e-5 (65B)
- Warmup: 2,000 steps
- LR decay: Cosine to 10% of max
- Weight decay: 0.1
- Gradient clipping: 1.0

### 13.2 LLaMA 2 (2023)

| Spec | 7B | 13B | 70B |
|------|-----|-----|-----|
| Layers | 32 | 40 | 80 |
| d_model | 4,096 | 5,120 | 8,192 |
| Heads | 32 | 40 | 64 |
| GQA groups | — | — | 8 |
| Context | 4,096 | 4,096 | 4,096 |
| Training tokens | 2.0T | 2.0T | 2.0T |

**Key changes:**
- **GQA** for 70B model (first mainstream use)
- Extended context (4K)
- Grouped attention for memory efficiency
- RLHF for chat versions

### 13.3 LLaMA 3 (2024)

| Spec | 8B | 70B | 405B |
|------|-----|-----|------|
| Layers | 32 | 80 | 126 |
| d_model | 4,096 | 8,192 | 16,384 |
| Heads | 32 | 64 | 128 |
| GQA groups | 8 | 8 | 8 |
| d_ff | 14,336 | 28,672 | 53,248 |
| Context | 128K | 128K | 128K |
| Vocabulary | 128,256 | 128,256 | 128,256 |
| Position | RoPE | RoPE | RoPE |
| Norm | Pre-RMSNorm | Pre-RMSNorm | Pre-RMSNorm |
| FFN | SwiGLU | SwiGLU | SwiGLU |
| Attention | GQA | GQA | GQA |
| Training tokens | 15T+ | 15T+ | 15T+ |

**Key innovations:**
- 128K context (vs 4K in LLaMA 2)
- 128K vocabulary (vs 32K)
- 15T+ training tokens (vs 2T)
- New special tokens for tool use
- Post-training: SFT + RLHF + rejection sampling

**405B training details:**
- 16K H100 GPUs
- Data parallel: 8
- Model parallel: 16
- Pipeline parallel: not used (fits on 16K GPUs)
- Batch size: 15.6M tokens
- LR: 8e-5
- Training time: ~54 days

---

## 14. DeepSeek FAMILY

### 14.1 DeepSeek-V2 (2024)

| Spec | Value |
|------|-------|
| Total parameters | 236B |
| Active parameters | 21B |
| Layers | 64 |
| d_model | 5,120 |
| Heads | 128 |
| d_head | 128 |
| MLA compressed dim | 512 |
| d_ff (per expert) | 2,048 |
| Context | 128K |
| Vocabulary | 100,000 |
| Position | RoPE |
| Norm | Pre-RMSNorm |
| FFN | SwiGLU (MoE) |
| Attention | MLA |
| MoE experts | 64 routed + 2 shared |
| Top-k | 6 |
| Training tokens | — |
| License | MIT |

**MLA Architecture Detail:**
```
For each layer:
  h → W_DQ (5,120 → 512) → c_Q
  h → W_DKV (5,120 → 512) → c_KV  [CACHED]

  Q = c_Q @ W_UQ (512 → 16,384) → reshape to 128 heads × 128 dim
  K = c_KV @ W_UK (512 → 16,384)
  V = c_KV @ W_UV (512 → 16,384)

  Attention(Q, K, V) → output
```

**KV cache per token:**
- MHA baseline: 2 × 64 layers × 128 heads × 128 dim × 2 bytes = 4.2 MB
- MLA: 64 layers × 512 dim × 2 bytes = 65.5 KB
- **Compression ratio: ~64×**

### 14.2 DeepSeek-V3 (2024)

| Spec | Value |
|------|-------|
| Total parameters | 671B |
| Active parameters | ~37B |
| Layers | 61 |
| d_model | 7,168 |
| Heads | 128 |
| d_head | 128 |
| MLA compressed dim | 512 |
| d_ff (per expert) | 2,048 |
| Context | 128K |
| Vocabulary | 128,000 |
| Position | RoPE |
| Norm | Pre-RMSNorm |
| FFN | SwiGLU (MoE) |
| Attention | MLA |
| MoE experts | 256 routed + 1 shared |
| Top-k | 8 |
| MTP heads | 2 |
| Training tokens | 14.8T |
| FP8 training | Yes |
| License | MIT |

**Aux-loss-free load balancing:**
```python
# Instead of auxiliary loss, use bias terms
bias = zeros(n_experts)

for step in training:
    # Forward pass
    scores = logits + bias
    selected = topk(softmax(scores), k)

    # Update bias (not backpropagated)
    expert_load = count(selected, n_experts)
    for i in range(n_experts):
        if expert_load[i] > target_load:
            bias[i] -= delta
        else:
            bias[i] += delta
```

**Multi-Token Prediction:**
```
Main output → predicts token t+1
MTP head 1 → predicts token t+2
MTP head 2 → predicts token t+3

Loss = Loss(t+1) + 0.3·Loss(t+2) + 0.2·Loss(t+3)
```

**Training cost:** ~$5.6M (2.8M H800 GPU hours)

---

## 15. Kimi FAMILY (Moonshot AI)

### 15.1 Kimi K2 (July 2025)

| Spec | Value |
|------|-------|
| Total parameters | 1.04T |
| Active parameters | 32B |
| Layers | 61 |
| d_model | 8,192 |
| Heads | 64 |
| d_head | 128 |
| MLA compressed dim | 512 |
| d_ff (per expert) | 2,048 |
| Context | 128K → 256K |
| Vocabulary | 160,000 |
| Position | RoPE + YaRN |
| Norm | Pre-RMSNorm |
| FFN | SwiGLU (MoE) |
| Attention | MLA + GQA hybrid |
| MoE experts | 384 routed + 1 shared |
| Top-k | 8 |
| Optimizer | Muon + QK-Clip |
| Training tokens | 15.5T |
| License | Modified MIT |

**Muon + QK-Clip detail:**
```python
# Muon for most layers
for param in model.parameters():
    if param.dim() == 2:  # Only 2D matrices
        param = muon_step(param, grad, lr)
    else:
        param = adamw_step(param, grad, lr)

# QK-Clip for attention stability
W_q, W_k = qk_clip(W_q, W_k, max_norm=1.0)
```

**Result:** Zero loss spikes during 1T parameter training.

### 15.2 Kimi K2.5 (January 2026)

| Spec | Value |
|------|-------|
| Base | Kimi K2 |
| Vision | MoonViT-3D (native) |
| Visual tokens | +15T mixed visual/text |
| Agent Swarm | Up to 100 sub-agents |
| Training | PARL (Parallel Agent RL) |
| Reward | 80% quality + 20% efficiency |

**MoonViT-3D:**
- Native vision encoder (not bolted on)
- 3D positional encoding for spatial understanding
- Processes images at multiple resolutions

### 15.3 Kimi K3 (July 2026)

| Spec | Value |
|------|-------|
| Total parameters | 2.8T |
| Active parameters | TBD |
| Architecture | MoE |
| Context | 1M tokens |
| Multimodal | Native (text, images, video) |
| Sparse attention | MoBA (deployed) |
| License | Open weights planned |

**Key capabilities:**
- Outperforms Claude and GPT on coding benchmarks
- "Vision in the loop" — iterates between code and live screenshots
- 2.8T parameters makes it the largest open-weight model

### 15.4 Kimi Linear (October 2025)

| Spec | Value |
|------|-------|
| Total parameters | 48B |
| Active parameters | 3B |
| Attention | KDA (Kimi Delta Attention) |
| Context | Long (optimized for KDA) |
| Speed | Faster than full attention at long contexts |

**KDA Architecture:**
```python
def kda_attention(Q, K, V):
    # Feature map (kernel trick)
    phi_Q = feature_map(Q)  # e.g., elu(Q) + 1
    phi_K = feature_map(K)

    # O(N·d²) computation via associativity
    KV = phi_K.T @ V        # (d, d)
    Z = phi_K.sum(dim=0)    # (d,)

    O = phi_Q @ KV          # (N, d)
    norm = phi_Q @ Z        # (N,)

    return O / norm.unsqueeze(-1)
```

**FlashKDA:** Open-source CUDA kernels optimized for KDA.

---

## 16. GLM FAMILY (Zhipu AI / z.ai)

### 16.1 GLM-4.5 (September 2025)

| Spec | Flagship | Air |
|------|----------|-----|
| Total parameters | 355B | 106B |
| Active parameters | 32B | 12B |
| Layers | 62 | 62 |
| d_model | 8,192 | 5,120 |
| Heads | 160 | 96 |
| d_head | 64 | 64 |
| Context | 128K | 128K |
| Vocabulary | 100,000 | 100,000 |
| Position | Partial RoPE | Partial RoPE |
| Norm | Pre-RMSNorm + QK-Norm | Pre-RMSNorm + QK-Norm |
| FFN | SwiGLU (MoE) | SwiGLU (MoE) |
| Attention | GQA (2.5× more heads) | GQA |
| MoE gates | Sigmoid | Sigmoid |
| MTP | MoE layers as MTP | MoE layers as MTP |
| Training tokens | 23T | 23T |
| License | MIT | MIT |

**Partial RoPE:**
- Only apply RoPE to a subset of dimensions
- Remaining dimensions use absolute position or no position
- Reduces computational overhead

**MoE layers as MTP:**
```
Standard MTP: Add separate prediction heads
GLM-4.5 MTP: The MoE layers themselves predict multiple tokens

Layer N (MoE):   processes token t
  → outputs representation for t
  → ALSO predicts token t+1 (via shared output projection)

This means speculative decoding is built into the architecture.
```

**Post-training pipeline:**
1. **Expert Training:** Create specialist models (coding, math, reasoning)
2. **Unified Training:** Integrate specialists via self-distillation

### 16.2 GLM-5.2 (June 2026)

| Spec | Value |
|------|-------|
| Total parameters | 753B |
| Active parameters | 40B |
| Layers | 64 |
| d_model | 8,192 |
| Heads | 160 |
| d_head | 64 |
| Context | 1M tokens |
| Vocabulary | 100,000 |
| Position | RoPE + long-context scaling |
| Norm | Pre-RMSNorm + QK-Norm |
| FFN | SwiGLU (DSA + MoE) |
| Attention | Sparse (IndexShare) |
| Sparse attention | IndexShare (every 4 layers share indexer) |
| MTP | Improved (+20% acceptance) |
| Training tokens | 28.5T |
| License | MIT |

**IndexShare Algorithm:**
```python
def indexshare_attention(x, layers, indexer):
    # layers[0]: compute indexer + sparse attention
    indices = indexer(x)  # top-k indices for sparse attention
    x = sparse_attention(x, indices)

    # layers[1-3]: reuse indices
    for layer in layers[1:4]:
        x = sparse_attention(x, indices)  # reuse same indices!

    return x
```

**Performance:** 2.9× FLOP reduction at 1M context vs traditional sparse attention.

**Long-horizon capabilities:**
- Sustains 1,700+ autonomous agent steps
- 8-hour continuous task execution
- Two thinking modes: High (fast) and Max (deep)

---

## 17. Qwen FAMILY

### 17.1 Qwen3 (May 2025)

| Spec | Dense | MoE |
|------|-------|-----|
| Max parameters | 32B | 235B |
| Architecture | Dense | MoE |
| Layers | 64 | 80 |
| d_model | 5,120 | 8,192 |
| Heads | 40 | 64 |
| Context | 128K | 128K |
| Vocabulary | 151,936 | 151,936 |
| Position | RoPE | RoPE |
| Norm | Pre-RMSNorm | Pre-RMSNorm |
| FFN | SwiGLU | SwiGLU (MoE) |
| Attention | GQA | GQA |
| Training tokens | ~20T | ~20T |
| License | Apache 2.0 | Apache 2.0 |

**Key feature:** Both dense and MoE variants released simultaneously.

---

## 18. Mistral/Mixtral FAMILY

### 18.1 Mistral 7B (2023)

| Spec | Value |
|------|-------|
| Parameters | 7.3B |
| Layers | 32 |
| d_model | 4,096 |
| Heads | 32 |
| d_head | 128 |
| d_ff | 14,336 |
| Context | 8K (sliding window) → 32K (with Flash Attention) |
| Vocabulary | 32,000 |
| Position | RoPE |
| Norm | Pre-RMSNorm |
| FFN | SwiGLU |
| Attention | GQA |
| Sliding window | 4,096 |

**Sliding Window Attention:**
```python
def sliding_window_attention(Q, K, V, window_size=4096):
    # Each token only attends to previous window_size tokens
    for i in range(seq_len):
        start = max(0, i - window_size)
        Q_i = Q[i:i+1]
        K_window = K[start:i+1]
        V_window = V[start:i+1]

        scores = Q_i @ K_window.T / sqrt(d_k)
        attn = softmax(scores)
        output[i] = attn @ V_window
```

**Performance:** Outperforms LLaMA 2 13B and approaches LLaMA 1 34B.

### 18.2 Mixtral 8×7B (2023)

| Spec | Value |
|------|-------|
| Total parameters | 46.7B |
| Active parameters | 12.9B (2 experts × 7B) |
| Architecture | Sparse MoE |
| Experts | 8 |
| Top-k | 2 |
| Context | 32K |

**Architecture:** 8 Mistral 7B models as experts, sparse routing.

---

## 19. Gemma/Gemini FAMILY

### 19.1 Gemma 2 (2024)

| Spec | 2B | 9B | 27B |
|------|-----|-----|-----|
| Layers | 26 | 42 | 46 |
| d_model | 2,304 | 3,584 | 4,608 |
| Heads | 8 | 16 | 32 |
| d_head | 256 | 256 | 256 |
| d_ff | 18,432 | 28,672 | 36,864 |
| Context | 128K | 128K | 128K |
| Vocabulary | 256,000 | 256,000 | 256,000 |
| Position | RoPE | RoPE | RoPE |
| Norm | Pre-RMSNorm + QK-Norm | Pre-RMSNorm + QK-Norm | Pre-RMSNorm + QK-Norm |
| FFN | GeGLU | GeGLU | GeGLU |
| Attention | GQA | GQA | GQA |
| Knowledge distillation | Yes (from Gemini) | Yes | Yes |

**Knowledge distillation:**
- Teacher: Gemini Ultra / Pro
- Student: Gemma 2
- Distill logits during training
- Result: Small model with outsized capabilities

### 19.2 Gemini (Google)

**Closed models — limited public details.**

| Known Spec | Value |
|-----------|-------|
| Architecture | Dense (reportedly) |
| Multimodal | Native (text, image, audio, video) |
| Context | Up to 2M tokens (Gemini 1.5 Pro) |
| Training | Multimodal from the start |

---

## 20. Claude FAMILY (Anthropic)

**Closed models — architecture details not public.**

| Model | Year | Known Capabilities |
|-------|------|-------------------|
| Claude 1 | 2023 | Constitutional AI, 100K context |
| Claude 2 | 2023 | 200K context, code, reasoning |
| Claude 3 (Haiku/Sonnet/Opus) | 2024 | Multimodal, 200K context |
| Claude 3.5 | 2024 | Artifacts, computer use, reasoning |
| Claude 4.8 | 2026 | Frontier coding, long-horizon tasks |

**Constitutional AI:**
1. Train model to critique its own outputs
2. Train model to revise based on constitutional principles
3. RLHF with AI feedback (not just human)

---

## 21. State Space Models (Mamba)

### 21.1 Mamba-1 (2023)

**Paper:** Mamba: Linear-Time Sequence Modeling with Selective State Spaces

**Core Idea:** Replace attention with a state space model that:
- Runs in O(N) time (not O(N²))
- Uses O(1) memory per step during inference
- Has input-dependent state transitions ("selective")

**State Space Model:**

$$
h'(t) = Ah(t) + Bx(t)
$$

$$
y(t) = Ch(t) + Dx(t)
$$

**Discretization (for digital implementation):**

$$
h_k = \bar{A}h_{k-1} + \bar{B}x_k
$$

$$
y_k = \bar{C}h_k + \bar{D}x_k
$$

**Mamba's innovation:** Make $\bar{B}$ and $\bar{C}$ input-dependent:

$$
\bar{B}_k = s_B(x_k), \quad \bar{C}_k = s_C(x_k)
$$

where $s_B, s_C$ are small linear projections.

**Pseudocode:**
```python
def mamba_block(x, A, D, W_b, W_c, W_d, conv1d):
    # x: (batch, seq, d_model)

    # 1. Short convolution (local context)
    x_conv = conv1d(x)
    x_conv = silu(x_conv)

    # 2. Selective SSM
    B = x_conv @ W_b  # (batch, seq, d_state)
    C = x_conv @ W_c  # (batch, seq, d_state)

    # 3. Discretized state space (parallel scan)
    y = selective_scan(x_conv, A, B, C, D)

    # 4. Gating
    return x + y * silu(x @ W_d)
```

**Selective Scan:**
```python
def selective_scan(x, A, B, C, D):
    # Parallel associative scan
    # h_k = A * h_{k-1} + B_k * x_k
    # Can be computed in parallel using associative operators

    # 1. Compute transition matrices
    # 2. Parallel scan (Blelloch scan)
    # 3. Compute outputs

    return y
```

**Complexity:**
- Training: O(N·d·d_state) — linear in sequence length!
- Inference: O(d·d_state) per step — constant in sequence length!

**Comparison with Transformer:**

| Aspect | Transformer | Mamba |
|--------|-------------|-------|
| Training time | O(N²·d) | O(N·d·d_state) |
| Inference memory | O(N·d) | O(d_state) |
| Global context | Yes (all pairs) | Yes (via state) |
| Gradient flow | Can be unstable | Stable |
| Look-up tasks | Excellent | Moderate |

### 21.2 Mamba-2 (2024)

**Paper:** Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality

**Key insight:** Certain attention patterns and certain SSM patterns are mathematically equivalent.

**SSD (Structured State Space Duality):**

The attention matrix can be written as:

$$
A_{ij} = \begin{cases} \alpha_i \alpha_{i-1} \cdots \alpha_{j+1} & \text{if } j < i \\ 0 & \text{otherwise} \end{cases}
$$

This is a **structured matrix** — it has a specific form that allows fast computation.

**Result:** Mamba-2 can use the same GPU optimizations as FlashAttention while maintaining linear complexity.

**Speedup:** 2-8× faster training than Mamba-1.

### 21.3 Mamba-3 (2025–2026)

**Reported innovations:**
- Complex-valued states (instead of real-valued)
- Multi-input multi-output (MIMO) state channels
- Better accuracy without slower decoding

---

## 22. Hybrid Architectures

### 22.1 Jamba (AI21, 2024)

| Spec | Value |
|------|-------|
| Architecture | Transformer + Mamba + MoE |
| Layers | 12 Transformer + 4 Mamba (repeated) |
| Context | 256K |
| MoE | 16 experts, top-2 |

**Pattern:**
```
[Transformer] → [Transformer] → [Transformer] → [Mamba] → [MoE]
(repeat)
```

### 22.2 Nemotron-H (NVIDIA, 2025)

| Spec | Value |
|------|-------|
| Architecture | 92% Mamba-2 + 8% Attention |
| Speedup | 3× vs LLaMA-3.1 |
| Accuracy | Same as LLaMA-3.1 |

### 22.3 Bamba (IBM, 2025)

| Spec | Value |
|------|-------|
| Architecture | Hybrid SSM + Transformer |
| Speedup | 2× throughput |
| Data efficiency | 7× less data than LLaMA-3.1-8B |

### 22.4 Hunyuan TurboS (Tencent, 2025)

| Spec | Value |
|------|-------|
| Architecture | Attention + Mamba-2 + MoE |
| Parameters | 560B total |
| Context | 256K |

---

# PART III: MATHEMATICAL APPENDIX

## A. Complete Attention Derivation

### A.1 Self-Attention as Kernel Regression

Attention can be viewed as kernel regression with a softmax kernel:

$$
\text{Attn}(Q, K, V)_i = \frac{\sum_{j=1}^{n} \exp(q_i^T k_j / \sqrt{d_k}) v_j}{\sum_{j=1}^{n} \exp(q_i^T k_j / \sqrt{d_k})}
$$

This is Nadaraya-Watson kernel regression with kernel $K(q, k) = \exp(q^T k / \sqrt{d_k})$.

### A.2 Gradient Flow Through Attention

**Backward pass for attention:**

Given $\frac{\partial L}{\partial O}$, compute $\frac{\partial L}{\partial Q}, \frac{\partial L}{\partial K}, \frac{\partial L}{\partial V}$.

$$
\frac{\partial L}{\partial V} = A^T \frac{\partial L}{\partial O}
$$

$$
\frac{\partial L}{\partial A} = \frac{\partial L}{\partial O} V^T
$$

$$
\frac{\partial L}{\partial S} = A \odot \left(\frac{\partial L}{\partial A} - A \cdot \text{diag}\left(\mathbf{1}^T \frac{\partial L}{\partial A}\right)\right)
$$

$$
\frac{\partial L}{\partial Q} = \frac{1}{\sqrt{d_k}} \frac{\partial L}{\partial S} K^T, \quad \frac{\partial L}{\partial K} = \frac{1}{\sqrt{d_k}} \left(\frac{\partial L}{\partial S}\right)^T Q
$$

### A.3 RoPE Derivation (Full)

**2D rotation matrix:**

$$
R_{\theta} = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}
$$

**Property:** $R_{\theta}$ is orthogonal: $R_{\theta}^T R_{\theta} = I$.

**For position $m$ and frequency $\theta$:**

$$
R_{m\theta} = \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix}
$$

**Key identity:**

$$
R_{m\theta}^T R_{n\theta} = R_{(n-m)\theta}
$$

**Proof:**

$$
R_{m\theta}^T R_{n\theta} = R_{-m\theta} R_{n\theta} = R_{(n-m)\theta}
$$

**Application to dot product:**

$$
(R_{m\theta} q)^T (R_{n\theta} k) = q^T R_{m\theta}^T R_{n\theta} k = q^T R_{(n-m)\theta} k
$$

This depends only on $n-m$ — the relative position!

### A.4 SwiGLU Derivation

**GLU (Gated Linear Unit):**

$$
\text{GLU}(x) = (xW) \odot \sigma(xV)
$$

**SwiGLU:** Replace sigmoid with Swish:

$$
\text{SwiGLU}(x) = \text{Swish}_1(xW) \odot xV
$$

where $\text{Swish}_1(x) = x \cdot \sigma(x)$.

**Why it works:**
- The "gate" $\sigma(xV)$ controls information flow
- Swish is smooth (unlike ReLU) and self-gated
- Element-wise product creates non-linearity

### A.5 MoE Routing Gradient

**Top-k routing is non-differentiable!**

Solution: Use straight-through estimator or soft top-k.

**Straight-through:**
```python
# Forward: hard top-k
indices = topk(scores, k)

# Backward: treat as if soft weights were used
weights = softmax(scores)
grad = grad_output * weights
```

**Load balancing gradient:**

$$
\frac{\partial \mathcal{L}_{\text{aux}}}{\partial W_g} = \alpha \cdot N \cdot \sum_{i=1}^{N} (f_i \cdot \frac{\partial P_i}{\partial W_g} + P_i \cdot \frac{\partial f_i}{\partial W_g})
$$

## B. Complexity Analysis

### B.1 Transformer Complexity

| Operation | Training Time | Training Memory | Inference Time | Inference Memory |
|-----------|--------------|-----------------|----------------|------------------|
| Embedding | O(N·d) | O(V·d) | O(N·d) | O(V·d) |
| Attention | O(N²·d) | O(N² + N·d) | O(N²·d) | O(N² + N·d) |
| FFN | O(N·d·d_ff) | O(N·d_ff) | O(N·d·d_ff) | O(N·d_ff) |
| Norm | O(N·d) | O(d) | O(N·d) | O(d) |
| **Total per layer** | **O(N²·d + N·d·d_ff)** | **O(N² + N·d_ff)** | **O(N²·d + N·d·d_ff)** | **O(N² + N·d_ff)** |
| **L layers** | **O(L·(N²·d + N·d·d_ff))** | **O(L·(N² + N·d_ff))** | **O(L·N²·d)** | **O(L·N²)** |

### B.2 MoE Complexity

| Aspect | Dense | MoE |
|--------|-------|-----|
| Forward FLOPs | O(N·d·d_ff) | O(N·d·d_ff·k/N_experts) |
| Active params | All | ~5-10% |
| Router overhead | None | O(N·d·N_experts) |
| Communication | None | All-to-all (expert parallel) |

### B.3 Mamba Complexity

| Aspect | Transformer | Mamba |
|--------|-------------|-------|
| Training time | O(N²·d) | O(N·d·d_state) |
| Training memory | O(N²) | O(N·d_state) |
| Inference time/step | O(N·d) | O(d·d_state) |
| Inference memory | O(N·d) | O(d_state) |

## C. Memory Layouts

### C.1 KV Cache Layout

**Standard layout (per layer):**
```
[K_cache]: (batch, seq_len, n_kv_heads, d_head)
[V_cache]: (batch, seq_len, n_kv_heads, d_head)
```

**MLA layout (per layer):**
```
[c_kv_cache]: (batch, seq_len, d_c)  # d_c << n_heads * d_head
```

### C.2 Weight Layout for Tensor Parallelism

**Column-wise split (for MLP):**
```
W1: (d_model, d_ff) → split into (d_model, d_ff/TP)
W2: (d_ff, d_model) → split into (d_ff/TP, d_model)
```

**Row-wise split (for attention):**
```
W_q: (d_model, d_model) → split into (d_model/TP, d_model)
```

### C.3 Expert Parallel Layout

```
GPU 0: Experts 0-15
GPU 1: Experts 16-31
GPU 2: Experts 32-47
GPU 3: Experts 48-63

All-to-all: Send tokens to GPU holding their selected expert
```

---

# References

## Foundational Papers

1. **Attention Is All You Need** — Vaswani et al., NeurIPS 2017 — [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
2. **BERT** — Devlin et al., NAACL 2019 — [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
3. **Improving Language Understanding by Generative Pre-Training** — Radford et al., 2018
4. **Language Models are Unsupervised Multitask Learners** — Radford et al., 2019
5. **Language Models are Few-Shot Learners** — Brown et al., NeurIPS 2020 — [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)
6. **Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer** — Raffel et al., JMLR 2020 — [arXiv:1910.10683](https://arxiv.org/abs/1910.10683)
7. **Switch Transformers** — Fedus et al., JMLR 2022 — [arXiv:2101.03961](https://arxiv.org/abs/2101.03961)

## Scaling & Efficiency

8. **Training Compute-Optimal Large Language Models** — Hoffmann et al., 2022 — [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)
9. **PaLM: Scaling Language Modeling with Pathways** — Chowdhery et al., 2022 — [arXiv:2204.02311](https://arxiv.org/abs/2204.02311)
10. **FlashAttention: Fast and Memory-Efficient Exact Attention** — Dao et al., NeurIPS 2022 — [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
11. **FlashAttention-2** — Dao, 2023 — [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)
12. **FlashAttention-3** — Shah et al., 2024 — [arXiv:2407.08608](https://arxiv.org/abs/2407.08608)

## Modern Open Models

13. **LLaMA: Open and Efficient Foundation Language Models** — Touvron et al., 2023 — [arXiv:2302.13971](https://arxiv.org/abs/2302.13971)
14. **Llama 2: Open Foundation and Fine-Tuned Chat Models** — Touvron et al., 2023 — [arXiv:2307.09288](https://arxiv.org/abs/2307.09288)
15. **The Llama 3 Herd of Models** — Meta AI, 2024 — [ai.meta.com](https://ai.meta.com/research/publications/the-llama-3-herd-of-models/)
16. **Mistral 7B** — Jiang et al., 2023 — [arXiv:2310.06825](https://arxiv.org/abs/2310.06825)
17. **Gemma: Open Models Based on Gemini** — Google, 2024 — [arXiv:2403.08295](https://arxiv.org/abs/2403.08295)

## DeepSeek

18. **DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model** — DeepSeek-AI, 2024 — [arXiv:2405.04434](https://arxiv.org/abs/2405.04434)
19. **DeepSeek-V3 Technical Report** — DeepSeek-AI, 2024 — [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)
20. **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning** — DeepSeek-AI, 2025 — [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)

## Moonshot AI

21. **MoBA: Mixture of Block Attention** — Moonshot AI, 2025 — [arXiv:2502.13189](https://arxiv.org/abs/2502.13189) | [GitHub](https://github.com/MoonshotAI/MoBA)
22. **Muon: An optimizer for hidden layers in neural networks** — Moonshot AI, 2025 — [arXiv:2502.16982](https://arxiv.org/abs/2502.16982) | [GitHub](https://github.com/MoonshotAI/Moonlight)
23. **FlashKDA** — Moonshot AI, 2025 — [GitHub](https://github.com/MoonshotAI/FlashKDA)
24. **Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving** — Moonshot AI, FAST 2025 — [arXiv:2407.00079](https://arxiv.org/abs/2407.00079)

## Zhipu AI / z.ai

25. **GLM-4.5 Technical Report** — Zhipu AI, 2025
26. **GLM-5.2** — Zhipu AI, 2026 — [z.ai](https://www.z.ai/) | [HuggingFace](https://huggingface.co/zai-org)

## State Space Models

27. **Mamba: Linear-Time Sequence Modeling with Selective State Spaces** — Gu & Dao, 2023 — [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)
28. **Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality** — Dao & Gu, 2024 — [arXiv:2405.21060](https://arxiv.org/abs/2405.21060)

## Key Algorithms

29. **RoFormer: Enhanced Transformer with Rotary Position Embedding** — Su et al., 2021 — [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
30. **Root Mean Square Layer Normalization** — Zhang & Sennrich, 2019 — [arXiv:1910.07467](https://arxiv.org/abs/1910.07467)
31. **GLU Variants Improve Transformer** — Shazeer, 2020 — [arXiv:2002.05202](https://arxiv.org/abs/2002.05202)
32. **GQA: Training Generalized Multi-Query Transformer Models** — Ainslie et al., 2023 — [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
33. **YaRN: Efficient Context Window Extension** — Peng et al., 2023 — [arXiv:2309.00071](https://arxiv.org/abs/2309.00071)
34. **ALiBi: Press et al., 2021** — [arXiv:2108.12409](https://arxiv.org/abs/2108.12409)
35. **Speculative Decoding** — Leviathan et al., 2022 — [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)
36. **DeepNorm** — Wang et al., 2022 — [arXiv:2203.00555](https://arxiv.org/abs/2203.00555)
37. **μP: Tensor Programs V** — Yang et al., 2022 — [arXiv:2203.03466](https://arxiv.org/abs/2203.03466)

## Additional References

38. **RetNet** — Sun et al., 2023 — [arXiv:2307.08621](https://arxiv.org/abs/2307.08621)
39. **RWKV** — Peng et al., 2023 — [arXiv:2305.13048](https://arxiv.org/abs/2305.13048)
40. **Hyena** — Poli et al., 2023 — [arXiv:2302.10866](https://arxiv.org/abs/2302.10866)
41. **Titans: Learning to Memorize at Test Time** — Bagnell et al., 2024 — [arXiv:2501.00663](https://arxiv.org/abs/2501.00663)
42. **Ring Attention** — Liu et al., 2024 — [arXiv:2310.01889](https://arxiv.org/abs/2310.01889)
43. **Infini-Attention** — Munkhdalai et al., 2024 — [arXiv:2401.04536](https://arxiv.org/abs/2401.04536)
44. **LongNet** — Ding et al., 2023 — [arXiv:2307.02486](https://arxiv.org/abs/2307.02486)
45. **BitNet** — Ma et al., 2024 — [arXiv:2402.17764](https://arxiv.org/abs/2402.17764)
46. **Sophia: A Scalable Stochastic Second-order Optimizer** — Liu et al., 2023 — [arXiv:2305.14342](https://arxiv.org/abs/2305.14342)
47. **Medusa** — Cai et al., 2024 — [arXiv:2401.10774](https://arxiv.org/abs/2401.10774)
48. **EAGLE** — Li et al., 2024 — [arXiv:2401.15077](https://arxiv.org/abs/2401.15077)

---

*This encyclopedia was compiled in July 2026. For the latest updates, check model cards on HuggingFace, official technical reports, and arXiv preprints.*


---

# PART IV: ADVANCED ARCHITECTURE TOPICS

---

## 23. Reasoning Architectures

### 23.1 Chain-of-Thought (CoT) Training

**Not an architecture change per se, but a training paradigm that affects architecture design.**

**Training data format:**
```
Question: What is 23 × 47?
Answer: Let's think step by step.
  23 × 40 = 920
  23 × 7 = 161
  920 + 161 = 1081
  Therefore, 23 × 47 = 1081.
```

**Architectural implications:**
- Longer context windows needed for reasoning traces
- Special tokens for reasoning start/end
- Test-time compute scaling (longer generation = better reasoning)

### 23.2 DeepSeek-R1 Architecture

**Paper:** DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (2025)

**Base model:** DeepSeek-V3 (671B MoE)

**Training pipeline:**
```
DeepSeek-V3-Base
  ↓
Cold Start: SFT on thousands of high-quality CoT examples
  ↓
RL Stage 1: GRPO (Group Relative Policy Optimization)
  ↓
Rejection Sampling: Generate SFT data from RL checkpoint
  ↓
RL Stage 2: General RL (reasoning + helpfulness + harmlessness)
  ↓
DeepSeek-R1
```

**GRPO (Group Relative Policy Optimization):**

Instead of using a separate value model (like PPO), GRPO uses group sampling:

```python
def grpo_step(policy, old_policy, question, group_size=8):
    # Sample group_size answers from old policy
    answers = [old_policy.generate(question) for _ in range(group_size)]

    # Compute rewards (e.g., correctness, format)
    rewards = [compute_reward(ans) for ans in answers]

    # Compute advantage as deviation from group mean
    mean_reward = mean(rewards)
    std_reward = std(rewards)
    advantages = [(r - mean_reward) / std_reward for r in rewards]

    # Update policy
    for ans, adv in zip(answers, advantages):
        ratio = policy.prob(ans) / old_policy.prob(ans)
        loss = -min(ratio * adv, clip(ratio, 0.8, 1.2) * adv)
        loss.backward()
```

**Key innovation:** No separate critic/value model needed. Reduces memory by ~50%.

**Self-evolution:** The model's own outputs improve over RL iterations. Reasoning patterns emerge spontaneously:
- Self-verification
- Backtracking
- Alternative approach exploration

### 23.3 Test-Time Compute Scaling

**Core idea:** Spend more compute at inference time to get better answers.

**Methods:**

| Method | How it works | Compute cost |
|--------|-------------|--------------|
| **CoT** | Generate reasoning steps | 2-10× tokens |
| **Self-consistency** | Sample N answers, majority vote | N× |
| **Tree of Thoughts** | Explore multiple reasoning paths | Variable |
| **Process Reward Model** | Reward intermediate steps | +PRM inference |
| **Monte Carlo Tree Search** | Search over reasoning space | Very high |

**Architectural implication:** Models need:
- Very long context windows (for long reasoning traces)
- Efficient inference (since we're generating more tokens)
- Special tokens for reasoning control

---

## 24. Multimodal Architectures

### 24.1 Vision Encoder + LLM (Early Approach)

```
Image → Vision Encoder (ViT) → Projection → LLM
                                    ↑
Text → Token Embedding ─────────────┘
```

**Examples:**
- **CLIP + GPT:** CLIP vision encoder, projected into GPT embedding space
- **LLaVA:** ViT-L/14 + LLaMA, trained with instruction tuning
- **BLIP-2:** Q-Former bridges vision and language

**Limitation:** Vision and language are "bolted together," not truly integrated.

### 24.2 Native Multimodal (Modern Approach)

**Kimi K3 / Gemini approach:**

```
Raw pixels / audio waveform / text tokens
  ↓
Unified tokenizer (handles all modalities)
  ↓
Native multimodal transformer
  ↓
Unified output (text + image generation)
```

**MoonViT-3D (Kimi K2.5+):**

```python
def moonvit_3d(image_patches):
    # 3D positional encoding: (x, y, scale)
    pos_3d = compute_3d_positions(image_patches)

    # Vision transformer with 3D-RoPE
    for layer in vision_layers:
        x = layer.norm(x)
        x = layer.attention(x, pos_3d)  # 3D-aware attention
        x = layer.ffn(x)

    # Project to LLM embedding space
    return projection(x)
```

**Key difference:** The vision encoder is trained jointly with the language model, not separately.

### 24.3 Interleaved Training

**Training data format:**
```
[Image 1] Text about image 1
[Image 2] Text about image 2, referencing image 1
[Image 3] Text about image 3
...
```

**Benefit:** Model learns relationships between images and text naturally.

---

## 25. Agentic Architectures

### 25.1 Tool Use as Architecture

**Modern models have special tokens for tool calls:**

```python
# LLaMA 3 tool tokens
<|start_header_id|>tool<|end_header_id|>
{"name": "calculator", "parameters": {"expression": "23*47"}}
<|eot_id|>

# Model generates tool call, system executes it, result fed back
<|start_header_id|>tool_response<|end_header_id|>
1081
<|eot_id|>
```

**GLM-4.5 tool format:**
```python
{
  "tool_calls": [{
    "id": "call_123",
    "type": "function",
    "function": {
      "name": "search",
      "arguments": '{"query": "latest AI papers"}'
    }
  }]
}
```

### 25.2 Agent Swarm (Kimi K2.5)

**Architecture:**
```
Orchestrator Agent (Kimi K2.5)
  ├── Sub-agent 1 (Research)
  ├── Sub-agent 2 (Coding)
  ├── Sub-agent 3 (Testing)
  ├── Sub-agent 4 (Documentation)
  └── ... up to 100 agents
```

**PARL (Parallel Agent Reinforcement Learning):**

```python
def parl_reward(task_result, execution_trace):
    quality_score = evaluate_quality(task_result)      # 80% weight
    efficiency_score = evaluate_efficiency(execution_trace)  # 20% weight
    return 0.8 * quality_score + 0.2 * efficiency_score
```

**Key insight:** The reward function explicitly optimizes for both correctness AND speed. This prevents agents from taking unnecessarily long reasoning paths.

### 25.3 Long-Horizon Task Architecture (GLM-5.2)

**Designed for tasks spanning hours:**

```
User request
  ↓
Planning module → Generate task graph
  ↓
Execution loop (up to 1,700+ steps):
  ├─ Execute step
  ├─ Observe result
  ├─ Update plan
  └─ Continue or backtrack
  ↓
Final result
```

**Thinking modes:**
- **High:** Fast responses, minimal reasoning
- **Max:** Deep chain-of-thought, self-verification, backtracking

---

## 26. Distributed Training Architectures

### 26.1 Data Parallelism (DP)

```
GPU 0: Model copy + Batch 0 → Gradients
GPU 1: Model copy + Batch 1 → Gradients
GPU 2: Model copy + Batch 2 → Gradients
GPU 3: Model copy + Batch 3 → Gradients
              ↓
         All-Reduce (average gradients)
              ↓
         All GPUs update model
```

**Limitation:** Each GPU holds full model. For 100B+ params, this doesn't fit.

### 26.2 Tensor Parallelism (TP)

**Split individual layers across GPUs.**

**For attention:**
```
GPU 0: Q_proj[0:d/2], K_proj[0:d/2], V_proj[0:d/2]
GPU 1: Q_proj[d/2:d], K_proj[d/2:d], V_proj[d/2:d]

Forward: All-Gather activations
Backward: All-Gather gradients
```

**For FFN (column-wise):**
```
GPU 0: W1[:, 0:d_ff/2]
GPU 1: W1[:, d_ff/2:d_ff]

Forward: No communication (each GPU computes partial output)
Backward: All-Reduce gradients
```

**Communication:** All-Gather and All-Reduce operations.

### 26.3 Pipeline Parallelism (PP)

**Split model depth across GPUs.**

```
GPU 0: Layers 0-11
GPU 1: Layers 12-23
GPU 2: Layers 24-35
GPU 3: Layers 36-47
```

**Problem: Pipeline bubbles**

```
Time →
GPU 0: [Fwd 0] [Fwd 1] [Fwd 2] ...
GPU 1:         [Fwd 0] [Fwd 1] ...
GPU 2:                 [Fwd 0] ...
GPU 3:                         ...
       ↑ bubble (idle time)
```

**Solutions:**
- **GPipe:** Micro-batching (split batch into smaller chunks)
- **PipeDream:** Interleave forward and backward passes
- **1F1B (One Forward One Backward):** Standard for LLMs

### 26.4 Expert Parallelism (EP)

**MoE-specific. Distribute experts across GPUs.**

```
GPU 0: Experts 0-15
GPU 1: Experts 16-31
GPU 2: Experts 32-47
GPU 3: Experts 48-63

All-to-all communication:
  GPU 0 sends tokens for experts 16-31 → GPU 1
  GPU 1 sends tokens for experts 0-15 → GPU 0
```

**Communication pattern:** All-to-all (every GPU talks to every other GPU).

**Optimization:**
- **Hierarchical all-to-all:** Within node first, then across nodes
- **Expert choice routing:** Let experts choose tokens (better load balance)

### 26.5 Sequence Parallelism (SP)

**Split sequence dimension across GPUs.**

```
GPU 0: Tokens 0-1023
GPU 1: Tokens 1024-2047
GPU 2: Tokens 2048-3071
GPU 3: Tokens 3072-4095
```

**Used for:** Very long sequences where even TP doesn't help.

**Ring Attention (Liu et al., 2024):**

```python
def ring_attention(Q, K, V, n_gpus):
    # Each GPU holds one block
    # Compute attention in a ring: each GPU passes K,V to next

    local_out = zeros_like(Q)
    local_max = full(-inf)
    local_sum = zeros()

    for step in range(n_gpus):
        # Receive K,V from previous GPU
        K_recv, V_recv = receive_from_prev()

        # Compute local attention block
        scores = Q @ K_recv.T
        local_max = max(local_max, max(scores))
        local_sum = local_sum * exp(old_max - local_max) + sum(exp(scores - local_max))
        local_out = local_out * exp(old_max - local_max) + exp(scores - local_max) @ V_recv

        # Send K,V to next GPU
        send_to_next(K_recv, V_recv)

    return local_out / local_sum
```

**Result:** Can train with context lengths limited only by number of GPUs.

### 26.6 3D Parallelism (DP + TP + PP)

**Combine all parallelism types:**

```
Data Parallel:      8 nodes
Tensor Parallel:    8 GPUs per node
Pipeline Parallel:  4 stages per GPU

Total GPUs: 8 × 8 × 4 = 256
```

**Memory breakdown for 1T model:**

| Component | Memory |
|-----------|--------|
| Model parameters (FP16) | 2TB |
| Optimizer states (Adam, FP32) | 4TB |
| Gradients (FP32) | 2TB |
| Activations | Variable |
| **Total per GPU (256-way)** | **~32GB** |

### 26.7 ZeRO (Zero Redundancy Optimizer)

**Paper:** Rajbhandari et al., 2020 (Microsoft)

**Idea:** Don't replicate optimizer states across all GPUs.

**Stages:**
- **ZeRO-1:** Partition optimizer states
- **ZeRO-2:** Partition optimizer states + gradients
- **ZeRO-3:** Partition optimizer states + gradients + parameters

**Result:** Can train models 8× larger on same hardware.

---

## 27. Memory Optimization Techniques

### 27.1 Activation Checkpointing

**Trade compute for memory.**

```python
# Without checkpointing
activations = []
x = input
for layer in layers:
    x = layer(x)
    activations.append(x)  # Store for backward

# With checkpointing
x = input
for i, layer in enumerate(layers):
    if i % checkpoint_interval == 0:
        x = checkpoint(layer, x)  # Recompute in backward
    else:
        x = layer(x)
```

**Memory savings:** ~50% for interval=2
**Compute overhead:** ~33% extra forward passes

### 27.2 Gradient Accumulation

**Simulate large batch sizes with limited memory.**

```python
accumulation_steps = 4
optimizer.zero_grad()

for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Effective batch size:** batch_size × accumulation_steps × num_gpus

### 27.3 Mixed Precision Training

**FP16/FP8 forward/backward, FP32 master weights.**

```python
# FP16 forward
with autocast(dtype=torch.float16):
    loss = model(input)

# FP16 backward
scaler.scale(loss).backward()

# FP32 optimizer step
scaler.step(optimizer)
scaler.update()
```

**FP8 (DeepSeek-V3):**
```python
# E4M3 for forward, E5M2 for backward
with autocast(dtype=torch.float8_e4m3fn):
    output = layer(input)
```

**Speedup:** 2× memory, 2× speed on supported hardware (H100).

### 27.4 Offloading

**Move optimizer states to CPU/NVMe.**

```
GPU: Model parameters + current batch
CPU: Optimizer states
NVMe: Checkpoints
```

**Used for:** Training very large models on limited GPU memory.

---

## 28. Inference Optimization

### 28.1 Continuous Batching

**Dynamic batching of incoming requests.**

```python
# Static batching (bad)
Batch 1: [req1(10 tokens), req2(100 tokens)] → wait for req2

# Continuous batching (good)
Time 0: [req1(10 tokens), req2(100 tokens)]
Time 1: [req1 done, req2(99 tokens), req3(50 tokens)]
Time 2: [req2(98 tokens), req3(49 tokens), req4(20 tokens)]
```

**Throughput improvement:** 10-20× for real-world request distributions.

### 28.2 PagedAttention

**Paper:** vLLM (Kwon et al., 2023)

**Problem:** KV cache memory is fragmented and pre-allocated.

**Solution:** Manage KV cache like OS virtual memory.

```python
# Traditional: pre-allocate max_seq_len for each request
kv_cache = zeros(batch_size, max_seq_len, n_layers, n_kv_heads, d_head)

# PagedAttention: allocate blocks on demand
block_size = 16  # tokens per block
kv_cache = BlockTable()  # Maps logical blocks to physical blocks

# When sequence grows:
if logical_block_full:
    allocate_new_physical_block()
```

**Benefits:**
- No memory waste from over-allocation
- Memory sharing (copy-on-write for parallel sampling)
- 2-4× throughput improvement

### 28.3 Speculative Decoding Variants

| Variant | Draft Model | Speedup | Complexity |
|---------|-------------|---------|------------|
| **Standard** | Small separate model | 2-3× | Medium |
| **Medusa** | Extra heads on same model | 2-3× | Low |
| **EAGLE** | Trained auto-regressive draft | 3-4× | High |
| **MTP (built-in)** | Model predicts multiple tokens | 2-3× | Lowest |
| **Lookahead Decoding** | Jacobi iteration | 2× | Medium |

**Medusa detail:**
```python
# Add multiple prediction heads to the same model
base_output = model.transformer(x)

token_t1 = lm_head(base_output)           # Standard next token
token_t2 = medusa_head_1(base_output)     # Predict t+2
token_t3 = medusa_head_2(base_output)     # Predict t+3

# During inference: verify all predictions in parallel
```

---

## 29. Quantization Architectures

### 29.1 GPTQ

**Post-training quantization to 4-bit.**

```python
def gptq_quantize(W, bits=4, group_size=128):
    # 1. Compute Hessian H = X^T X
    H = compute_hessian(activations)

    # 2. For each column (in order):
    for i in range(W.shape[1]):
        # Quantize column
        w_q = round(W[:, i] / scale) * scale

        # Compute quantization error
        err = W[:, i] - w_q

        # Update remaining weights (optimal brain surgeon)
        W[:, i+1:] -= err[:, None] @ H[i, i+1:][None, :] / H[i, i]

    return W_q
```

**Result:** 4-bit weights with <1% perplexity degradation.

### 29.2 AWQ (Activation-aware Weight Quantization)

**Idea:** Not all weights are equally important. Protect "salient" weights.

```python
def awq_quantize(W, bits=4):
    # Compute activation magnitudes
    activation_scale = abs(activations).mean(dim=0)

    # Identify salient channels (high activation magnitude)
    salient_channels = topk(activation_scale, k=int(0.1 * len(activation_scale)))

    # Quantize non-salient weights to 4-bit
    # Keep salient weights in higher precision (or scale them)
    W_quantized = quantize(W, bits=bits)
    W_quantized[:, salient_channels] *= scale_factor

    return W_quantized
```

### 29.3 BitNet

**Train in 1-bit from scratch.**

```python
def bitnet_linear(x, W):
    # Weights are 1-bit: {-1, +1}
    W_b = sign(W)

    # Scale factor per output channel
    alpha = mean(abs(W), dim=1)

    # Forward pass
    y = x @ W_b.T
    y = y * alpha  # Apply scale

    return y
```

**Result:** 1-bit weights with comparable quality to FP16 for some tasks.

---

## 30. Retrieval-Augmented Generation (RAG) Architectures

### 30.1 Standard RAG

```
Query → Dense Retriever (bi-encoder) → Top-k documents → Concatenate with query → LLM → Answer
```

**Retriever:**
```python
# Bi-encoder (separate encoders for query and document)
query_embedding = query_encoder(query)        # (d,)
doc_embeddings = doc_encoder(documents)         # (N, d)

# Similarity
scores = query_embedding @ doc_embeddings.T     # (N,)
top_k_docs = documents[topk(scores, k)]
```

### 30.2 RETRO (Retrieval-Enhanced Transformer)

**Paper:** Borgeaud et al., 2022 (DeepMind)

**Architecture:**
```
Input tokens
  ↓
[Chunked Encoder] → Retrieve neighbors for each chunk
  ↓
[Cross-attention with retrieved chunks]
  ↓
[Standard Transformer layers]
  ↓
Output
```

**Key idea:** Retrieve at the chunk level (not document level). Each 64-token chunk retrieves 2 nearest neighbors.

**Database:** 2 trillion tokens (The Pile + more)

### 30.3 Long-Context as Implicit Retrieval

**Modern approach (Gemini 1.5, Claude 3):**

Instead of explicit retrieval, just make the context window large enough to fit everything:

```
Entire codebase → Context window (1M-2M tokens)
Entire book → Context window
Entire conversation history → Context window
```

**Trade-off:**
- RAG: Cheaper, but retrieval quality matters
- Long context: Simpler, but more expensive at inference

---

## 31. Memory Architectures

### 31.1 Titans

**Paper:** Bagnell et al., 2024

**Core idea:** Neural memory module that learns to remember and forget.

```python
class TitansMemory:
    def __init__(self, dim):
        self.memory = zeros(dim, dim)  # Learned memory matrix
        self.forget_gate = Linear(dim, dim)
        self.remember_gate = Linear(dim, dim)

    def forward(self, x):
        # What to forget
        forget = sigmoid(self.forget_gate(x))
        self.memory = self.memory * forget

        # What to remember
        remember = tanh(self.remember_gate(x))
        self.memory = self.memory + remember.T @ x

        # Retrieve
        return x @ self.memory
```

### 31.2 Infini-Attention

**Paper:** Munkhdalai et al., 2024 (Google)

**Combines local attention with compressive memory:**

```python
def infini_attention(Q, K, V, local_window):
    # Local attention (standard)
    local_out = sliding_window_attention(Q, K, V, local_window)

    # Compressive memory (stores summary of all previous tokens)
    memory = compress(K, V)  # O(1) memory per step
    memory_out = retrieve(Q, memory)

    # Combine
    gate = sigmoid(combine_gate(Q))
    return gate * local_out + (1 - gate) * memory_out
```

**Result:** Infinite context length with bounded memory.

---

## 32. Hardware-Aware Architecture Design

### 32.1 Roofline Model

**Key insight for architecture design:**

```
Performance = min(Compute Peak, Memory Bandwidth × Arithmetic Intensity)
```

**For Transformers:**
- Attention: Memory-bound (low arithmetic intensity)
- FFN: Compute-bound (high arithmetic intensity)

**Design implication:**
- Make attention as memory-efficient as possible (FlashAttention, MLA)
- Make FFN as wide as possible (up to compute limit)

### 32.2 Communication-Aware MoE

**All-to-all communication is the bottleneck for MoE.**

**Optimizations:**
- **Hierarchical routing:** Route within node first, then across nodes
- **Expert capacity factor:** Limit tokens per expert to prevent overflow
- **Load balancing:** Ensure even distribution (aux loss or bias terms)

### 32.3 FlashAttention as Architecture Primitive

**FlashAttention is not just an optimization — it enables new architectures:**

- **Long context:** Without FlashAttention, 128K context is impractical
- **Ring Attention:** Enables training on unlimited context (given enough GPUs)
- **Sliding window:** Only possible with efficient attention kernels

---

## 33. Complete Paper Index

### Foundational (2017–2020)

| # | Paper | Authors | Year | Venue | Link |
|---|-------|---------|------|-------|------|
| 1 | Attention Is All You Need | Vaswani et al. | 2017 | NeurIPS | [arXiv](https://arxiv.org/abs/1706.03762) |
| 2 | Layer Normalization | Ba et al. | 2016 | — | [arXiv](https://arxiv.org/abs/1607.06450) |
| 3 | BERT: Pre-training of Deep Bidirectional Transformers | Devlin et al. | 2018 | NAACL | [arXiv](https://arxiv.org/abs/1810.04805) |
| 4 | Improving Language Understanding by Generative Pre-Training | Radford et al. | 2018 | — | [OpenAI](https://openai.com/research/language-unsupervised) |
| 5 | Language Models are Unsupervised Multitask Learners | Radford et al. | 2019 | — | [OpenAI](https://openai.com/research/better-language-models) |
| 6 | Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer | Raffel et al. | 2019 | JMLR | [arXiv](https://arxiv.org/abs/1910.10683) |
| 7 | RoBERTa: A Robustly Optimized BERT Pretraining Approach | Liu et al. | 2019 | — | [arXiv](https://arxiv.org/abs/1907.11692) |
| 8 | XLNet: Generalized Autoregressive Pretraining | Yang et al. | 2019 | NeurIPS | [arXiv](https://arxiv.org/abs/1906.08237) |
| 9 | Language Models are Few-Shot Learners | Brown et al. | 2020 | NeurIPS | [arXiv](https://arxiv.org/abs/2005.14165) |
| 10 | Switch Transformers: Scaling to Trillion Parameter Models | Fedus et al. | 2021 | JMLR | [arXiv](https://arxiv.org/abs/2101.03961) |

### Scaling & Efficiency (2021–2023)

| # | Paper | Authors | Year | Venue | Link |
|---|-------|---------|------|-------|------|
| 11 | Training Compute-Optimal Large Language Models | Hoffmann et al. | 2022 | — | [arXiv](https://arxiv.org/abs/2203.15556) |
| 12 | PaLM: Scaling Language Modeling with Pathways | Chowdhery et al. | 2022 | — | [arXiv](https://arxiv.org/abs/2204.02311) |
| 13 | Scaling Language Models: Methods, Analysis & Insights from Training Gopher | Rae et al. | 2022 | — | [arXiv](https://arxiv.org/abs/2112.11446) |
| 14 | GLaM: Efficient Scaling of Language Models with Mixture-of-Experts | Du et al. | 2022 | — | [arXiv](https://arxiv.org/abs/2112.06905) |
| 15 | FlashAttention: Fast and Memory-Efficient Exact Attention | Dao et al. | 2022 | NeurIPS | [arXiv](https://arxiv.org/abs/2205.14135) |
| 16 | FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning | Dao | 2023 | — | [arXiv](https://arxiv.org/abs/2307.08691) |
| 17 | FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision | Shah et al. | 2024 | — | [arXiv](https://arxiv.org/abs/2407.08608) |
| 18 | ZeRO: Memory Optimizations Toward Training Trillion Parameter Models | Rajbhandari et al. | 2020 | SC | [arXiv](https://arxiv.org/abs/1910.02054) |
| 19 | Efficient Large-Scale Language Model Training on GPU Clusters | Narayanan et al. | 2021 | SC | [arXiv](https://arxiv.org/abs/2104.04473) |

### Modern Open Models (2023–2024)

| # | Paper | Authors | Year | Venue | Link |
|---|-------|---------|------|-------|------|
| 20 | LLaMA: Open and Efficient Foundation Language Models | Touvron et al. | 2023 | — | [arXiv](https://arxiv.org/abs/2302.13971) |
| 21 | Llama 2: Open Foundation and Fine-Tuned Chat Models | Touvron et al. | 2023 | — | [arXiv](https://arxiv.org/abs/2307.09288) |
| 22 | The Llama 3 Herd of Models | Meta AI | 2024 | — | [Meta](https://ai.meta.com/research/publications/the-llama-3-herd-of-models/) |
| 23 | Mistral 7B | Jiang et al. | 2023 | — | [arXiv](https://arxiv.org/abs/2310.06825) |
| 24 | Mixtral of Experts | Jiang et al. | 2024 | — | [arXiv](https://arxiv.org/abs/2401.04088) |
| 25 | Gemma: Open Models Based on Gemini | Google | 2024 | — | [arXiv](https://arxiv.org/abs/2403.08295) |
| 26 | Gemma 2: Improving Open Language Models at a Practical Size | Google | 2024 | — | [arXiv](https://arxiv.org/abs/2408.00118) |
| 27 | Qwen Technical Report | Bai et al. | 2023 | — | [arXiv](https://arxiv.org/abs/2309.16609) |
| 28 | Qwen2 Technical Report | Yang et al. | 2024 | — | [arXiv](https://arxiv.org/abs/2407.10671) |

### DeepSeek

| # | Paper | Authors | Year | Venue | Link |
|---|-------|---------|------|-------|------|
| 29 | DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model | DeepSeek-AI | 2024 | — | [arXiv](https://arxiv.org/abs/2405.04434) |
| 30 | DeepSeek-V3 Technical Report | DeepSeek-AI | 2024 | — | [arXiv](https://arxiv.org/abs/2412.19437) |
| 31 | DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning | DeepSeek-AI | 2025 | — | [arXiv](https://arxiv.org/abs/2501.12948) |

### Moonshot AI

| # | Paper | Authors | Year | Venue | Link |
|---|-------|---------|------|-------|------|
| 32 | MoBA: Mixture of Block Attention | Moonshot AI | 2025 | ICLR | [arXiv](https://arxiv.org/abs/2502.13189) |
| 33 | Muon: An optimizer for hidden layers in neural networks | Moonshot AI | 2025 | — | [arXiv](https://arxiv.org/abs/2502.16982) |
| 34 | Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving | Moonshot AI | 2025 | FAST | [arXiv](https://arxiv.org/abs/2407.00079) |
| 35 | FlashKDA | Moonshot AI | 2025 | — | [GitHub](https://github.com/MoonshotAI/FlashKDA) |

### Zhipu AI / z.ai

| # | Paper | Authors | Year | Venue | Link |
|---|-------|---------|------|-------|------|
| 36 | GLM: General Language Model Pretraining with Autoregressive Blank Infilling | Du et al. | 2022 | ACL | [arXiv](https://arxiv.org/abs/2103.10360) |
| 37 | ChatGLM: A Family of Large Language Models from GLM-130B to GLM-4 All Tools | Zhipu AI | 2024 | — | [arXiv](https://arxiv.org/abs/2406.12793) |
| 38 | GLM-4.5 Technical Report | Zhipu AI | 2025 | — | [z.ai](https://www.z.ai/) |
| 39 | GLM-5.2 | Zhipu AI | 2026 | — | [z.ai](https://www.z.ai/) |

### State Space Models

| # | Paper | Authors | Year | Venue | Link |
|---|-------|---------|------|-------|------|
| 40 | Mamba: Linear-Time Sequence Modeling with Selective State Spaces | Gu & Dao | 2023 | — | [arXiv](https://arxiv.org/abs/2312.00752) |
| 41 | Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality | Dao & Gu | 2024 | — | [arXiv](https://arxiv.org/abs/2405.21060) |
| 42 | Jamba: A Hybrid Transformer-Mamba Language Model | Lieber et al. | 2024 | — | [arXiv](https://arxiv.org/abs/2403.19887) |
| 43 | Nemotron-H | NVIDIA | 2025 | — | [NVIDIA](https://www.nvidia.com/) |
| 44 | Bamba: Simple Hybrid Mamba-Transformer | IBM | 2025 | — | [arXiv](https://arxiv.org/abs/2502.18906) |

### Key Algorithms

| # | Paper | Authors | Year | Venue | Link |
|---|-------|---------|------|-------|------|
| 45 | RoFormer: Enhanced Transformer with Rotary Position Embedding | Su et al. | 2021 | — | [arXiv](https://arxiv.org/abs/2104.09864) |
| 46 | Root Mean Square Layer Normalization | Zhang & Sennrich | 2019 | — | [arXiv](https://arxiv.org/abs/1910.07467) |
| 47 | GLU Variants Improve Transformer | Shazeer | 2020 | — | [arXiv](https://arxiv.org/abs/2002.05202) |
| 48 | GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints | Ainslie et al. | 2023 | — | [arXiv](https://arxiv.org/abs/2305.13245) |
| 49 | Multi-Query Attention is All You Need | Shazeer | 2019 | — | [arXiv](https://arxiv.org/abs/1911.02150) |
| 50 | YaRN: Efficient Context Window Extension of Large Language Models | Peng et al. | 2023 | — | [arXiv](https://arxiv.org/abs/2309.00071) |
| 51 | ALiBi: Press et al. | 2021 | — | [arXiv](https://arxiv.org/abs/2108.12409) |
| 52 | Speculative Decoding: Leveraging Speculative Execution for Accelerating LLM Inference | Leviathan et al. | 2022 | — | [arXiv](https://arxiv.org/abs/2211.17192) |
| 53 | Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads | Cai et al. | 2024 | — | [arXiv](https://arxiv.org/abs/2401.10774) |
| 54 | EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty | Li et al. | 2024 | — | [arXiv](https://arxiv.org/abs/2401.15077) |
| 55 | DeepNorm: A Pre-Normalization Method for Deep Transformer Networks | Wang et al. | 2022 | — | [arXiv](https://arxiv.org/abs/2203.00555) |
| 56 | μP: Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer | Yang et al. | 2022 | — | [arXiv](https://arxiv.org/abs/2203.03466) |
| 57 | RetNet: Retentive Network: A Successor to Transformer for Large Language Models | Sun et al. | 2023 | — | [arXiv](https://arxiv.org/abs/2307.08621) |
| 58 | RWKV: Reinventing RNNs for the Transformer Era | Peng et al. | 2023 | — | [arXiv](https://arxiv.org/abs/2305.13048) |
| 59 | Hyena Hierarchy: Towards Larger Convolutional Language Models | Poli et al. | 2023 | — | [arXiv](https://arxiv.org/abs/2302.10866) |
| 60 | Titans: Learning to Memorize at Test Time | Bagnell et al. | 2024 | — | [arXiv](https://arxiv.org/abs/2501.00663) |
| 61 | Ring Attention with Blockwise Transformers for Near-Infinite Context | Liu et al. | 2024 | — | [arXiv](https://arxiv.org/abs/2310.01889) |
| 62 | Infini-attention: Infinite Context with Compressible Memory | Munkhdalai et al. | 2024 | — | [arXiv](https://arxiv.org/abs/2401.04536) |
| 63 | LongNet: Scaling Transformers to 1,000,000,000 Tokens | Ding et al. | 2023 | — | [arXiv](https://arxiv.org/abs/2307.02486) |
| 64 | BitNet: Scaling 1-bit Transformers for Large Language Models | Ma et al. | 2024 | — | [arXiv](https://arxiv.org/abs/2402.17764) |
| 65 | Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training | Liu et al. | 2023 | — | [arXiv](https://arxiv.org/abs/2305.14342) |
| 66 | vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention | Kwon et al. | 2023 | SOSP | [arXiv](https://arxiv.org/abs/2309.06180) |
| 67 | GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers | Frantar et al. | 2022 | — | [arXiv](https://arxiv.org/abs/2210.17323) |
| 68 | AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration | Lin et al. | 2023 | — | [arXiv](https://arxiv.org/abs/2306.00978) |
| 69 | LoRA: Low-Rank Adaptation of Large Language Models | Hu et al. | 2021 | ICLR | [arXiv](https://arxiv.org/abs/2106.09685) |
| 70 | QLoRA: Efficient Finetuning of Quantized LLMs | Dettmers et al. | 2023 | NeurIPS | [arXiv](https://arxiv.org/abs/2305.14314) |
| 71 | InstructGPT: Training language models to follow instructions with human feedback | Ouyang et al. | 2022 | NeurIPS | [arXiv](https://arxiv.org/abs/2203.02155) |
| 72 | Constitutional AI: Harmlessness from AI Feedback | Bai et al. | 2022 | — | [arXiv](https://arxiv.org/abs/2212.08073) |
| 73 | RLHF: A General Framework for Learning from Human Feedback | Various | 2023 | — | Multiple papers |
| 74 | DPO: Direct Preference Optimization | Rafailov et al. | 2023 | NeurIPS | [arXiv](https://arxiv.org/abs/2305.18290) |
| 75 | KTO: Model Alignment as Prospect Theoretic Optimization | Ethayarajh et al. | 2024 | — | [arXiv](https://arxiv.org/abs/2402.01306) |
| 76 | RETRO: Improving language models by retrieving from trillions of tokens | Borgeaud et al. | 2022 | NeurIPS | [arXiv](https://arxiv.org/abs/2112.04426) |
| 77 | UL2: Unifying Language Learning Paradigms | Tay et al. | 2022 | — | [arXiv](https://arxiv.org/abs/2205.05131) |
| 78 | Flamingo: a Visual Language Model for Few-Shot Learning | Alayrac et al. | 2022 | NeurIPS | [arXiv](https://arxiv.org/abs/2204.14198) |
| 79 | BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models | Li et al. | 2023 | — | [arXiv](https://arxiv.org/abs/2301.12597) |
| 80 | LLaVA: Visual Instruction Tuning | Liu et al. | 2023 | — | [arXiv](https://arxiv.org/abs/2304.08485) |

---

## 34. Hyperparameter Reference Tables

### 34.1 Learning Rate by Model Size

| Model Size | Base LR | Warmup | Min LR | Batch Size (tokens) |
|-----------|---------|--------|--------|---------------------|
| 125M | 6e-4 | 2,000 | 6e-5 | 0.5M |
| 350M | 3e-4 | 2,000 | 3e-5 | 0.5M |
| 1B | 3e-4 | 2,000 | 3e-5 | 1M |
| 7B | 3e-4 | 2,000 | 3e-5 | 4M |
| 13B | 1.5e-4 | 2,000 | 1.5e-5 | 4M |
| 30B | 1e-4 | 2,000 | 1e-5 | 4M |
| 65B | 8e-5 | 2,000 | 8e-6 | 4M |
| 175B | 6e-5 | 3,750 | 6e-6 | 3.2M |
| 405B | 8e-5 | 8,000 | 8e-6 | 15.6M |
| 671B (DeepSeek-V3) | 2.2e-4 | 2,000 | 2.2e-5 | 15.4M |

### 34.2 Model Dimension Scaling

| Model | d_model | n_layers | d_ff | n_heads | d_head |
|-------|---------|----------|------|---------|--------|
| GPT-3 Small | 768 | 12 | 3,072 | 12 | 64 |
| GPT-3 Medium | 1,024 | 24 | 4,096 | 16 | 64 |
| GPT-3 Large | 1,280 | 36 | 5,120 | 20 | 64 |
| GPT-3 XL | 1,600 | 48 | 6,400 | 25 | 64 |
| GPT-3 2.7B | 2,560 | 32 | 10,240 | 32 | 80 |
| GPT-3 6.7B | 4,096 | 32 | 16,384 | 32 | 128 |
| GPT-3 13B | 5,120 | 40 | 20,480 | 40 | 128 |
| GPT-3 175B | 12,288 | 96 | 49,152 | 96 | 128 |
| LLaMA-7B | 4,096 | 32 | 11,008 | 32 | 128 |
| LLaMA-13B | 5,120 | 40 | 13,824 | 40 | 128 |
| LLaMA-33B | 6,656 | 60 | 17,920 | 52 | 128 |
| LLaMA-65B | 8,192 | 80 | 22,016 | 64 | 128 |
| LLaMA-2-70B | 8,192 | 80 | 28,672 | 64 | 128 |
| LLaMA-3-8B | 4,096 | 32 | 14,336 | 32 | 128 |
| LLaMA-3-70B | 8,192 | 80 | 28,672 | 64 | 128 |
| LLaMA-3-405B | 16,384 | 126 | 53,248 | 128 | 128 |
| DeepSeek-V2 | 5,120 | 64 | 2,048 (per expert) | 128 | 128 |
| DeepSeek-V3 | 7,168 | 61 | 2,048 (per expert) | 128 | 128 |
| Kimi K2 | 8,192 | 61 | 2,048 (per expert) | 64 | 128 |
| GLM-4.5 | 8,192 | 62 | — | 160 | 64 |
| GLM-5.2 | 8,192 | 64 | — | 160 | 64 |

### 34.3 Context Length Evolution

| Year | Model | Context | Technique |
|------|-------|---------|-----------|
| 2017 | Transformer | 512 | — |
| 2018 | GPT-1 | 512 | — |
| 2019 | GPT-2 | 1,024 | — |
| 2020 | GPT-3 | 2,048 | — |
| 2022 | PaLM | 2,048 | — |
| 2022 | LLaMA | 2,048 | — |
| 2023 | LLaMA-2 | 4,096 | — |
| 2023 | Mistral | 8,000 → 32,000 | Sliding window + FlashAttention |
| 2023 | Claude 2 | 100,000 | — |
| 2024 | LLaMA-3 | 128,000 | RoPE scaling |
| 2024 | Gemini 1.5 Pro | 1,000,000 | — |
| 2024 | DeepSeek-V2 | 128,000 | MLA |
| 2025 | Kimi K2 | 256,000 | YaRN + MLA |
| 2025 | GLM-4.5 | 128,000 | — |
| 2026 | GLM-5.2 | 1,000,000 | IndexShare + DSA |
| 2026 | Kimi K3 | 1,000,000 | MoBA |

---

## 35. Glossary of Terms

| Term | Definition |
|------|-----------|
| **Attention** | Mechanism allowing tokens to "look at" each other |
| **Autoregressive** | Generating one token at a time, left-to-right |
| **BPE** | Byte-Pair Encoding, subword tokenization |
| **CoT** | Chain-of-Thought, step-by-step reasoning |
| **Decoder-only** | Transformer with only decoder (causal attention) |
| **Dense model** | All parameters active for every token |
| **d_model** | Hidden dimension of the model |
| **d_ff** | Feed-forward network hidden dimension |
| **Embedding** | Mapping from token IDs to dense vectors |
| **Encoder-decoder** | Transformer with both encoder and decoder |
| **Expert** | Sub-network in a Mixture of Experts |
| **FFN** | Feed-Forward Network |
| **FlashAttention** | IO-aware attention algorithm |
| **FP8** | 8-bit floating point precision |
| **GQA** | Grouped Query Attention |
| **KV cache** | Cached key and value vectors for inference |
| **LayerNorm** | Normalization across feature dimension |
| **LM Head** | Final linear layer projecting to vocabulary |
| **LoRA** | Low-Rank Adaptation, parameter-efficient fine-tuning |
| **MLA** | Multi-Head Latent Attention |
| **MLM** | Masked Language Modeling |
| **MoE** | Mixture of Experts |
| **MQA** | Multi-Query Attention |
| **MTP** | Multi-Token Prediction |
| **Pre-norm** | Normalization before sublayer |
| **Post-norm** | Normalization after sublayer |
| **RLHF** | Reinforcement Learning from Human Feedback |
| **RMSNorm** | Root Mean Square Normalization |
| **RoPE** | Rotary Position Embedding |
| **Router** | Network deciding which experts to use |
| **Self-attention** | Attention where Q, K, V come from same input |
| **Sliding window** | Attention limited to nearby tokens |
| **Softmax** | Normalization function: exp(x_i) / sum(exp(x_j)) |
| **Speculative decoding** | Draft model predicts multiple tokens, target verifies |
| **SSM** | State Space Model |
| **SwiGLU** | Swish-Gated Linear Unit activation |
| **Tensor parallelism** | Splitting layers across GPUs |
| **Tokenizer** | Converts text to token IDs |
| **TPU** | Tensor Processing Unit (Google AI accelerator) |
| **Vocabulary** | Set of all possible tokens |
| **Weight tying** | Sharing weights between input and output embeddings |
| **YaRN** | Yet another RoPE extension method |
| **ZeRO** | Zero Redundancy Optimizer |

---

*This encyclopedia represents the state of AI architecture as of July 2026. The field evolves rapidly — verify latest specifications from official model cards and technical reports.*
