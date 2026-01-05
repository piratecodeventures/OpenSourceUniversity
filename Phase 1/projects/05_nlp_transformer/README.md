# 🤖 NLP Project: The Universal Translator

> **Officer Kael's Log**: "We need to talk to them. I've built two communication devices. One runs on Torch energy, the other on Keras crystals. Both do the exact same math."

This project implements a **Sequence-to-Sequence Transformer** for Machine Translation.
To demonstrate mastery of the ecosystem, we implement the **Exact Same Architecture** in both **PyTorch** and **TensorFlow/Keras**.

## 🎯 The Task
**Toy Translation**: English $\to$ Alien.
*   Rule: Reverse the word order and uppercase vowels.
*   Input: "hello world"
*   Target: "DLRW LLH" (Simulated language).

## 🛠️ implementations

### 1. PyTorch (`model_torch.py`)
*   Uses `nn.Transformer`.
*   Manual Training Loop.
*   Data Loaders.

### 2. Keras (`model_keras.py`)
*   Uses `MultiHeadAttention` layer.
*   `model.fit()` training loop.
*   Functional API.

## 🚀 Usage

```bash
# Run PyTorch Version
python model_torch.py

# Run Keras Version
python model_keras.py
```

## 🧠 Key Concepts Demonstrated
1.  **Positional Encoding**: Injecting order into the sequence.
2.  **Masking**: Preventing the decoder from establishing causality with future tokens.
3.  **Cross-Entropy Loss**: Standard for language generation.
