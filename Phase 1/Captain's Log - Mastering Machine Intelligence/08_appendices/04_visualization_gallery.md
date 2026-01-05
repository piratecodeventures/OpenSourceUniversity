# Appendix D: Visualization Gallery

A collection of the systems diagrams used in this book.

## 🏗️ Neural Architectures

### The Perceptron
```mermaid
graph LR
    x1((x1)) -- w1 --> Sum((Σ))
    x2((x2)) -- w2 --> Sum
    b((b)) -- 1 --> Sum
    Sum --> Act[Activation f]
    Act --> y((y))
```

### The Transformer (Encoder-Decoder)
```mermaid
graph TD
    Input --> Embed[Embedding + PosEnc]
    Embed --> EncBlock[Encoder Block]
    EncBlock --> EncBlock
    EncBlock --> DecBlock[Decoder Block]
    TargetInput --> MaskEmbed[Masked Embed]
    MaskEmbed --> DecBlock
    DecBlock --> Softmax
    Softmax --> Output
```

## 🔄 Machine Learning Loops

### The Supervised Loop
```mermaid
graph LR
    Data --> Model
    Model --> Prediction
    Prediction -- vs Truth --> Loss
    Loss -- Gradient --> Optimizer
    Optimizer -- Update --> Model
```

### The Agent Loop (ReAct)
```mermaid
graph TD
    User --> Agent
    Agent -- "Thought" --> LLM
    LLM -- "Action" --> Tool
    Tool -- "Observation" --> Agent
    Agent -- "Update" --> LLM
    LLM -- "Final Answer" --> User
```

## 🚢 MLOps & Systems

### Docker Build Flow
```mermaid
graph LR
    Dev[Code] --> Build[Docker Build]
    Build --> Image[Image]
    Image --> Push[Registry]
    Push --> Pull[Server]
    Pull --> Run[Container]
```

### RAG System
```mermaid
graph LR
    User[Query] --> Embed[Embed Model]
    Embed --> Vec[Vector]
    Vec -- Search --> DB[Vector DB]
    DB -- Retrieve --> Context
    Context & Query --> Prompt
    Prompt --> LLM
    LLM --> Answer
```
