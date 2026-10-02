# Attention Is All You Need — Complete Beginner-Friendly Explanation

Paper: **“Attention Is All You Need”** by Ashish Vaswani and 7 co-authors, published in 2017. This is the paper that introduced the **Transformer architecture**, the basic architecture behind modern systems such as GPT-style language models, BERT, T5, and many other models. DOI

Read the original paper on arXiv

I’m going to explain it as if you are starting with **zero knowledge of machine learning, neural networks, vectors, matrices, attention, or NLP**.

The goal is not merely to tell you _what_ the Transformer does. We will build the ideas from the ground up and then walk through essentially every section of the paper.

---

# 0. First: What is this paper actually about?

Imagine you give a computer:

> **“I love cats.”**

and ask it to translate it into French:

> **“J'aime les chats.”**

A computer needs to learn a complicated relationship between the input sequence and output sequence.

Before this paper, one popular approach was to use something called an **RNN — Recurrent Neural Network**.

The basic idea of an RNN was:

```
word 1 → word 2 → word 3 → word 4 → ...
          ↓
        memory
```

It reads things sequentially.

The Transformer says:

> **What if we don't have to process words one after another?**

Instead, let every word look at every other word and decide:

> "Which other words are important to me?"

That mechanism is called **attention**.

And the Transformer essentially says:

> **Let's build the whole sequence model around attention instead of recurrence.**

That's the central idea.

The paper's architecture is called:

# Transformer

And its most important ingredients are:

1. Token embeddings
    
2. Positional encoding
    
3. Self-attention
    
4. Scaled dot-product attention
    
5. Multi-head attention
    
6. Feed-forward networks
    
7. Residual connections
    
8. Layer normalization
    
9. Encoder
    
10. Decoder
    
11. Masked attention
    
12. Softmax
    
13. Training with Adam
    
14. Dropout
    
15. Label smoothing
    
16. Beam search during translation
    

We'll understand every one.

---

# 1. Before the paper: What problem are they solving?

The paper is primarily about **sequence transduction**.

That sounds complicated, but it simply means:

> Take one sequence and transform it into another sequence.

For example:

### Translation

```
English:
I love cats.

        ↓ model

French:
J'aime les chats.
```

### Summarization

```
Long article
     ↓
Short summary
```

### Question answering

```
Question
     ↓
Answer
```

### Speech recognition

```
Audio sequence
     ↓
Text sequence
```

The Transformer paper demonstrates its architecture mainly on **machine translation** and also tests it on **English constituency parsing**. arXiv+1

---

# 2. The absolute basics: What is a neural network?

Before understanding Transformer, we need to understand what a neural network is.

Suppose we want a computer to determine whether an animal is a cat.

We could give it numbers representing properties:

```
ears = 1
fur = 1
tail = 1
wings = 0
```

A neural network takes numbers as input and performs mathematical operations on them.

A very simplified neuron might do:

y=w1x1+w2x2+w3x3+by = w_1x_1+w_2x_2+w_3x_3+b

where:

- xx = input
    
- ww = learned weights whose attention patterns appear involved in this an enormous amount of machinery into those pages. The explanation above covers its architecture, equations, motivation, computational comparison, training procedure, regularization, experiments, model variations, parsing experiment, attention visualizations, and conclusions.
    
- bb = bias
    
- yy = output
    

The network learns the weights.

For example:

```
ears ───────┐
            │
fur ────────┼──> neural network ──> cat probability
            │
tail ───────┘
```

During training, the network changes its weights until its predictions become better.

---

# 3. What does "learning" mean?

Suppose the correct answer is:

```
cat
```

but the model predicts:

```
dog
```

We calculate an **error**, usually called a **loss**.

The model then changes its internal parameters to reduce that loss.

Very roughly:

```
prediction
    ↓
calculate error
    ↓
calculate gradients
    ↓
change weights
    ↓
better prediction
```

This process is repeated many times.

---

# 4. What is a sequence?

A sequence is simply an ordered collection.

For example:

```
I → love → cats
```

Order matters.

Compare:

> "Dog bites man."

with:

> "Man bites dog."

Same words, completely different meaning.

Therefore a language model needs to know:

- which word exists
    
- what the word means
    
- where the word occurs
    
- what other words it relates to
    

This becomes extremely important for Transformers.

---

# 5. Why not just give words directly to a neural network?

Computers ultimately work with numbers.

A word like:

```
cat
```

needs to become numbers.

This is where **tokens** and **embeddings** come in.

---

# 6. What is a token?

A token is a piece of text that the model processes.

It could be:

- a whole word
    
- part of a word
    
- punctuation
    
- sometimes a special symbol
    

For example:

```
"I love cats!"
```

might become:

```
["I", "love", "cat", "s", "!"]
```

The exact tokenization method varies.

The original Transformer experiments use subword methods including **Byte Pair Encoding (BPE)** and word-piece representations. arXiv

---

# 7. What is an embedding?

Suppose we assign every token an ID:

```
I       → 17
love    → 928
cat     → 4312
```

These IDs aren't useful mathematically.

We therefore convert each token into a vector.

For example:

```
cat
 ↓
[0.21, -0.43, 0.72, 0.11, ...]
```

Maybe the vector has 512 numbers.

This is called an:

# Embedding

An embedding is basically:

> **A numerical representation of a token that the neural network can work with.**

The Transformer paper uses an embedding dimension of:

dmodel=512d_{model}=512

for the base model. arXiv+1

---

# 8. What is a vector?

Don't let the mathematical notation scare you.

A vector is simply a list of numbers.

Example:

[2,5,1][2,5,1]

That's a 3-dimensional vector.

A Transformer embedding might look conceptually like:

[0.12,−0.44,0.91,...][0.12,-0.44,0.91,...]

with 512 values.

You can think of it as a **coordinate describing the token in a mathematical space**.

---

# 9. What is a matrix?

A matrix is simply a table of numbers.

For example:

[123456]\begin{bmatrix} 1&2&3\\ 4&5&6 \end{bmatrix}

It has:

- 2 rows
    
- 3 columns
    

Transformers perform enormous numbers of matrix operations.

Why?

Because modern GPUs are extremely good at matrix multiplication.

This is one reason Transformers can be highly parallelized.

---

# 10. What was wrong with RNNs?

Now we arrive at the main motivation of the paper.

Before Transformers, **RNNs**, LSTMs, GRUs, etc. were extremely important sequence models.

Suppose we have:

> "The cat that was sitting on the mat was hungry."

An RNN processes:

```
The
 ↓
cat
 ↓
that
 ↓
was
 ↓
sitting
 ↓
on
 ↓
the
 ↓
mat
 ↓
was
 ↓
hungry
```

The next step depends on the previous step.

So:

ht=f(ht−1,xt)h_t=f(h_{t-1},x_t)

where:

- xtx_t = current input
    
- hth_t = current hidden state
    
- ht−1h_{t-1} = previous hidden state
    

This means:

> You cannot fully process word 10 before processing word 9.

That's **sequential computation**.

The paper identifies this as a major limitation because it makes training harder to parallelize. arXiv

---

# 11. Why is sequential processing bad?

Imagine 1,000 words.

An RNN essentially has a chain:

```
1 → 2 → 3 → 4 → ... → 999 → 1000
```

You have to move through that chain.

A Transformer can process many positions simultaneously:

```
1 ─┐
2 ─┤
3 ─┤
4 ─┤──> attention
5 ─┤
6 ─┤
...│
```

The GPU can perform many of these operations in parallel.

This was one of the major advantages identified by the paper.

---

# 12. Another problem: long-distance relationships

Consider:

> "The animal didn't cross the road because **it** was too tired."

What does **it** refer to?

Probably:

> animal

The model needs to connect words that might be far apart.

RNNs theoretically can remember distant information, but the computational path can become long.

The Transformer changes this.

With self-attention:

```
"it" ───────────────→ "animal"
```

It can directly connect one position to another.

The paper emphasizes that self-attention gives a **constant-length path** between positions, compared with O(n)O(n) sequential steps for recurrence. arXiv

---

# 13. What is attention?

This is the most important concept in the entire paper.

Imagine I ask:

> "The animal didn't cross the road because it was tired."

When processing:

> **it**

we want the model to ask:

> "Which other words should I pay attention to?"

Maybe:

```
The       0.01
animal    0.72  ← important
didn't    0.02
cross     0.03
the       0.01
road      0.08
because   0.02
it        0.05
was       0.03
tired     0.03
```

These numbers are called **attention weights**.

The model uses them to create a new representation of "it" using information from other tokens.

That's attention.

The paper defines attention as mapping:

- a query
    
- a collection of keys
    
- corresponding values
    

to an output. arXiv

---

# 14. Query, Key and Value

This terminology initially feels weird.

Let's use a library analogy.

Suppose you walk into a library looking for information.

You have a question:

> "Where are books about space?"

That's your:

# Query

Each book has information describing what it contains.

That's like a:

# Key

Once you find relevant books, you take the actual information from them.

That's the:

# Value

So:

```
Query
"What am I looking for?"

       ↓ compare

Keys
"What does each item contain?"

       ↓ choose relevant ones

Values
"Give me the actual information."
```

This is not a perfect analogy, but it is useful.

---

# 15. Attention in one sentence

A simple description is:

> **Attention lets one token decide which other tokens are important to it.**

That's the heart of the Transformer.

---

# 16. The mathematical definition of attention

The paper gives:

Attention(Q,K,V)=softmax(QKTdk)VAttention(Q,K,V) = softmax \left( \frac{QK^T}{\sqrt{d_k}} \right)V

This looks terrifying.

Let's break it into pieces.

---

# 17. Step 1: Q

QQ means:

# Query

It represents:

> "What information am I looking for?"

---

# 18. Step 2: K

KK means:

# Key

It represents:

> "What kind of information do I contain?"

---

# 19. Step 3: V

VV means:

# Value

It represents:

> "Here is the actual information you can use."

---

# 20. Step 4: QKTQK^T

This is a matrix multiplication.

The important idea is:

> We compare each query against every key.

Suppose we have:

```
Query: "it"

Keys:

The
animal
didn't
cross
road
it
was
tired
```

The model computes a score for:

```
it ↔ The
it ↔ animal
it ↔ didn't
it ↔ cross
...
```

If the score between:

```
it ↔ animal
```

is high, that means:

> "These two representations appear compatible."

---

# 21. What is the dot product?

Suppose:

q=[1,2,3]q=[1,2,3]

and:

k=[4,5,6]k=[4,5,6]

Their dot product is:

1(4)+2(5)+3(6)1(4)+2(5)+3(6)

=4+10+18=4+10+18

=32=32

A larger dot product generally indicates greater alignment between the vectors.

So attention uses dot products to calculate compatibility.

---

# 22. Why KTK^T?

If KK is a matrix containing many keys, we transpose it so the matrix dimensions line up for multiplication.

You don't need to memorize this as a mysterious Transformer trick.

It's ordinary linear algebra used to efficiently calculate:

> every query against every key.

---

# 23. Why divide by dk\sqrt{d_k}?

This is one of the details specifically introduced in the paper.

The attention score is:

QKTdk\frac{QK^T}{\sqrt{d_k}}

Why?

Because as the dimension dkd_k increases, dot products can become large.

Large numbers entering softmax can cause softmax to become extremely peaked.

Then gradients can become very small.

The authors explain that if the components of qq and kk are independent random variables with mean 0 and variance 1, the dot product has variance dkd_k. Therefore its magnitude tends to grow with dimension. arXiv

Dividing by:

dk\sqrt{d_k}

controls the scale.

Hence:

# Scaled Dot-Product Attention

---

# 24. What is softmax?

Suppose we have scores:

```
2.0
1.0
0.5
```

Softmax converts them into probabilities-like values:

```
0.63
0.23
0.14
```

They sum to 1.

So instead of saying:

```
animal = score 10
road = score 3
```

we can say approximately:

```
animal → 0.8
road   → 0.1
...
```

These become attention weights.

---

# 25. Why do we need softmax?

Because we want to turn raw compatibility scores into something that tells us:

> **How much should I use each value?**

For example:

```
animal  → 0.70
road    → 0.15
because → 0.05
other   → 0.10
```

Then we calculate a weighted combination of the values.

---

# 26. The whole attention calculation

The process is:

```
Q + K
 ↓
calculate similarity
 ↓
QKᵀ
 ↓
scale by √dk
 ↓
softmax
 ↓
attention weights
 ↓
multiply by V
 ↓
output
```

Mathematically:

QKTQK^T

then

QKTdk\frac{QK^T}{\sqrt{d_k}}

then

softmax(QKTdk)softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)

then:

softmax(QKTdk)Vsoftmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V

That's Equation 1 of the paper. arXiv

---

# 27. Why use matrices Q, K and V?

Instead of processing one word at a time, the Transformer puts many queries together in a matrix.

For example:

```
Q =
[query for word 1
 query for word 2
 query for word 3
 ...]
```

Likewise:

```
K =
[key for word 1
 key for word 2
 key for word 3
 ...]
```

and:

```
V =
[value for word 1
 value for word 2
 value for word 3
 ...]
```

Then the GPU can calculate attention for all positions together.

This is a major reason for the Transformer's parallelism.

---

# 28. Self-attention

Now we get to the key idea.

Suppose the input is:

> "The cat sat on the mat."

In **self-attention**, the:

- Queries
    
- Keys
    
- Values
    

all come from the **same sequence**.

So:

```
Input sequence
      ↓
 ┌────┼────┐
 Q    K     V
```

Every token can potentially attend to every other token.

The paper calls this **self-attention**, also sometimes called **intra-attention**. arXiv

---

# 29. Example of self-attention

Sentence:

> "The animal crossed the road because it was tired."

When processing:

> "it"

attention might learn something like:

```
The       0.01
animal    0.60
crossed   0.03
road      0.08
because   0.02
it        0.05
was       0.02
tired     0.19
```

The exact numbers are learned by the model.

The important point is:

> "it" can directly access information from "animal".

---

# 30. The Transformer architecture

Now we're ready for the main architecture.

The original Transformer has:

```
                TRANSFORMER

       ┌─────────────────────────┐
       │        ENCODER          │
Input ─┤                         │
       │  Layer 6                │
       │  Layer 5                │
       │  Layer 4                │
       │  Layer 3                │
       │  Layer 2                │
       │  Layer 1                │
       └───────────┬─────────────┘
                   │
                   ↓
       ┌─────────────────────────┐
       │        DECODER          │
       │  Layer 1                │
       │  Layer 2                │
       │  Layer 3                │
       │  Layer 4                │
       │  Layer 5                │
       │  Layer 6                │
       └───────────┬─────────────┘
                   ↓
                Output
```

The paper uses **6 encoder layers and 6 decoder layers** in the base configuration. arXiv

---

# 31. What does the encoder do?

The encoder receives the input.

For translation:

```
English sentence
       ↓
encoder
       ↓
rich representation
```

It tries to understand the input sequence.

---

# 32. What does the decoder do?

The decoder generates the output.

For translation:

```
English representation
        ↓
decoder
        ↓
French sentence
```

The decoder generates the output **one token at a time** during inference.

This is important:

> The Transformer removes recurrence from the architecture's internal processing, but the original decoder still generates output autoregressively.

---

# 33. What does autoregressive mean?

It sounds complicated but means:

> The model uses previously generated outputs to generate the next output.

Suppose we want:

> "I love cats."

The decoder might generate:

```
<START>
   ↓
I
   ↓
love
   ↓
cats
   ↓
<EOS>
```

When generating `love`, it knows `I`.

When generating `cats`, it knows:

```
I love
```

The paper explicitly says the decoder generates one element at a time and consumes previously generated symbols as additional input. arXiv

---

# 34. Inside one encoder layer

Each encoder layer contains two major pieces:

### 1. Multi-head self-attention

and

### 2. Feed-forward neural network

The paper says each encoder layer has these two sublayers, with residual connections and layer normalization around them. arXiv

Conceptually:

```
Input
  │
  ▼
Multi-Head Self-Attention
  │
  + ◄──── residual connection
  │
LayerNorm
  │
  ▼
Feed-Forward Network
  │
  + ◄──── residual connection
  │
LayerNorm
  │
Output
```

And this whole thing is repeated six times.

---

# 35. What is a residual connection?

Suppose the input is:

xx

and a subnetwork produces:

F(x)F(x)

Instead of only returning:

F(x)F(x)

we return:

x+F(x)x+F(x)

That's a:

# Residual connection

or:

# Skip connection

So information can bypass the complicated layer.

The paper uses:

LayerNorm(x+Sublayer(x))LayerNorm(x+Sublayer(x))

for each sublayer in the original architecture. arXiv

---

# 36. Why residual connections?

Deep neural networks can be difficult to train.

If you stack many layers:

```
Layer 1
 ↓
Layer 2
 ↓
Layer 3
 ↓
...
 ↓
Layer 50
```

information and gradients can become difficult to preserve.

Residual connections create an easier path:

```
───────────────┐
               ↓
x → complex layer → +
```

The network can learn:

> "I don't need to change this much."

instead of having to completely transform the representation every time.

---

# 37. What is Layer Normalization?

A neural network's internal numbers can have different scales.

Layer normalization helps stabilize these values.

Conceptually:

```
messy activations
       ↓
normalize
       ↓
more stable activations
```

The paper uses LayerNorm after the residual addition in its described architecture. arXiv

---

# 38. Multi-head attention

Now we reach another extremely important concept.

Why not just have one attention mechanism?

The paper found it useful to have:

# Multiple attention heads

The base Transformer uses:

h=8h=8

attention heads.

Each head works in a different learned representation subspace. arXiv

---

# 39. Why multiple heads?

Imagine reading:

> "The animal crossed the road because it was tired."

One attention mechanism might learn:

```
"it" → "animal"
```

Another might learn:

```
"tired" → "was"
```

Another might focus on:

```
"crossed" → "road"
```

Another might learn something related to grammatical structure.

We don't manually tell the heads to do these things.

They learn different relationships during training.

---

# 40. How multi-head attention works

Suppose:

dmodel=512d_{model}=512

and:

h=8h=8

Then each head uses:

dk=dv=64d_k=d_v=64

because:

512/8=64512/8=64

The paper explicitly uses eight heads with 64-dimensional key/value representations in the base Transformer. arXiv

---

# 41. Each head has its own projections

The model takes the original representation and creates:

```
Q → Q₁, Q₂, ..., Q₈

K → K₁, K₂, ..., K₈

V → V₁, V₂, ..., V₈
```

Each head has its own learned matrices.

For head ii:

headi=Attention(QWiQ,KWiK,VWiV)head_i = Attention(QW_i^Q,KW_i^K,VW_i^V)

Then:

```
head 1 ─┐
head 2 ─┤
head 3 ─┤
...     ├──> concatenate → linear projection
head 8 ─┘
```

The paper gives:

MultiHead(Q,K,V)=Concat(head1,…,headh)WOMultiHead(Q,K,V) = Concat(head_1,\ldots,head_h)W^O

arXiv

---

# 42. What are WQW^Q, WKW^K, WVW^V?

These are learned parameter matrices.

They transform the input representation into:

- query representation
    
- key representation
    
- value representation
    

For example:

Q=XWQQ=XW^Q

K=XWKK=XW^K

V=XWVV=XW^V

The model learns the values of these matrices during training.

---

# 43. Why reduce each head to 64 dimensions?

If we had eight full 512-dimensional attention mechanisms, computation would be huge.

Instead:

```
512 dimensions
      ↓
8 heads
      ↓
64 dimensions each
```

Then:

8×64=5128\times64=512

So the combined representation has the original dimension.

The paper notes that the reduced dimension per head keeps total computational cost similar to a single full-dimensional attention mechanism. arXiv

---

# 44. What happens after the heads?

Suppose each head produces 64 numbers.

Eight heads produce:

8×64=5128\times64=512

These are concatenated.

Then another learned matrix:

WOW^O

projects them back into the model dimension.

So:

```
Head 1 → 64
Head 2 → 64
Head 3 → 64
...
Head 8 → 64

       ↓

Concatenate

       ↓

512

       ↓

Linear projection

       ↓

512
```

---

# 45. Three different uses of attention in the Transformer

The paper explicitly describes **three uses**.

This is important.

## A. Encoder self-attention

Queries, keys, and values all come from the encoder.

```
Encoder representation
       ↓
 Q K V
       ↓
self-attention
```

Every input token can look at every input token.

---

## B. Decoder self-attention

Queries, keys and values come from the decoder.

But there is an important restriction.

A decoder position cannot look at future output tokens.

This is called:

# Masked self-attention

---

## C. Encoder-decoder attention

The decoder asks questions using its queries.

The encoder provides keys and values.

Conceptually:

```
Decoder → Query

Encoder → Keys + Values
```

This lets the decoder look back at the input sequence.

The paper describes all three mechanisms explicitly. arXiv

---

# 46. Why does the decoder need masking?

Suppose we're generating:

```
I love cats
```

When predicting:

```
love
```

the model should not be allowed to look at:

```
cats
```

because `cats` is the future answer.

Otherwise the model could cheat.

So we need:

```
Position 1 can see:
1

Position 2 can see:
1,2

Position 3 can see:
1,2,3

Position 4 can see:
1,2,3,4
```

but not:

```
Position 2 → Position 3
Position 2 → Position 4
```

---

# 47. The attention mask

The paper implements this by setting illegal attention scores to:

−∞-\infty

before softmax.

Why?

Because:

softmax(−∞)≈0softmax(-\infty)\approx0

So the model effectively assigns zero attention to future tokens.

The paper explicitly describes this masking mechanism. arXiv

---

# 48. Decoder architecture

Each decoder layer contains **three** sublayers:

### 1. Masked multi-head self-attention

### 2. Encoder-decoder multi-head attention

### 3. Feed-forward network

Each gets residual connections and LayerNorm.

Conceptually:

```
Input from previous decoder layer
          │
          ▼
Masked Self-Attention
          │
          + residual
          │
       LayerNorm
          │
          ▼
Encoder-Decoder Attention
          │
          + residual
          │
       LayerNorm
          │
          ▼
Feed Forward Network
          │
          + residual
          │
       LayerNorm
          │
          ▼
Output
```

Six of these decoder layers are stacked in the base Transformer. arXiv

---

# 49. The feed-forward network

Attention isn't the only thing happening.

Every encoder and decoder layer also contains a:

# Position-wise Feed-Forward Network

The paper gives:

FFN(x)=max(0,xW1+b1)W2+b2FFN(x) = max(0,xW_1+b_1)W_2+b_2

This is Equation 2. arXiv

---

# 50. What is ReLU?

The paper uses:

ReLU(x)=max(0,x)ReLU(x)=max(0,x)

So:

```
x = -5 → 0
x = -2 → 0
x =  0 → 0
x =  3 → 3
x =  7 → 7
```

It introduces non-linearity.

---

# 51. Why is the feed-forward network called "position-wise"?

Because the same neural network is applied separately to every token position.

Suppose:

```
I       → vector
love    → vector
cats    → vector
```

The FFN processes:

```
I       → FFN
love    → FFN
cats    → FFN
```

independently.

The **same weights** are used for each position within that layer.

The paper describes this explicitly. arXiv

---

# 52. Dimensions of the FFN

The model dimension is:

dmodel=512d_{model}=512

The internal FFN dimension is:

dff=2048d_{ff}=2048

So conceptually:

```
512
 ↓
2048
 ↓
512
```

The paper uses exactly these dimensions for the base model. arXiv

---

# 53. Why expand from 512 to 2048?

Think of it like giving the neural network a larger workspace.

It takes a 512-dimensional representation and transforms it into a 2048-dimensional hidden representation.

Then it compresses it back to 512.

```
small representation
       ↓
larger workspace
       ↓
processed representation
```

The attention mechanism mixes information **between tokens**.

The feed-forward network then performs more complex processing **within each token representation**.

This distinction is very useful.

---

# 54. Another way to understand an encoder layer

Think of each encoder layer as doing two jobs:

### Attention:

> "Which other words should I look at?"

### Feed-forward:

> "Now that I've gathered this information, what should I compute from it?"

Then repeat.

```
look around
    ↓
think/process
    ↓
look around again
    ↓
think/process again
    ↓
...
```

---

# 55. Embeddings

At the bottom of the model, tokens must become vectors.

The paper uses learned embeddings to convert tokens into vectors of dimension:

dmodeld_{model}

The same general idea is used for output tokens. arXiv

---

# 56. Why multiply embeddings by dmodel\sqrt{d_{model}}?

The paper says:

> In the embedding layers, the weights are multiplied by dmodel\sqrt{d_{model}}.

For the base model:

512≈22.63\sqrt{512}\approx22.63

This is part of the scaling used in their architecture.

You don't need to interpret this as "making the word mean more."

It's a numerical scaling choice intended to keep representation magnitudes in a useful range relative to other components.

---

# 57. The huge missing problem: order

Here's an extremely important problem.

Self-attention by itself doesn't inherently know word order.

Consider:

> "Dog bites man."

and:

> "Man bites dog."

If we only had a bag of word vectors, we would have:

```
Dog
bites
man
```

in both cases.

But the order changes the meaning.

RNNs automatically process things in sequence.

The Transformer deliberately removed recurrence.

So:

# How does Transformer know position?

Answer:

# Positional Encoding

---

# 58. Positional encoding

The Transformer adds another vector to each word embedding.

Conceptually:

```
word meaning vector
       +
position vector
       =
input representation
```

For example:

```
"cat"
   +
"position 3"
   ↓
final representation
```

---

# 59. The paper's positional encoding formula

The paper uses sine and cosine functions.

For even dimensions:

PE(pos,2i)=sin⁡(pos100002i/dmodel)PE(pos,2i) = \sin\left( \frac{pos}{10000^{2i/d_{model}}} \right)

For odd dimensions:

PE(pos,2i+1)=cos⁡(pos100002i/dmodel)PE(pos,2i+1) = \cos\left( \frac{pos}{10000^{2i/d_{model}}} \right)

arXiv

Don't panic.

The important idea is:

> Different dimensions use different frequencies of sine/cosine waves.

---

# 60. Why sine and cosine?

Imagine drawing waves.

```
dimension 1:
~~~~~~~ ~~~~~~~

dimension 2:
~~ ~~ ~~ ~~ ~~

dimension 3:
~ ~ ~ ~ ~ ~ ~
```

Different dimensions change at different rates.

Together, they create a unique-ish mathematical signature for each position.

So position 1 gets one pattern.

Position 2 gets another.

Position 100 gets another.

---

# 61. Why not simply use position = 1, 2, 3?

Because the model works with vectors.

A scalar:

```
position = 7
```

doesn't naturally provide the rich representation they wanted.

Instead, positional encoding gives each position a vector.

---

# 62. Why did the authors choose sinusoidal encoding?

They hypothesized that it could help the model learn relative positions.

The paper says that for a fixed offset kk, the encoding at position pos+kpos+k can be represented as a linear function of the encoding at pospos. arXiv

They also tested learned positional embeddings.

Interestingly, the results were nearly identical.

They chose sinusoidal encoding partly because they believed it might allow extrapolation to sequence lengths longer than those seen during training. arXiv

---

# 63. Where is positional encoding added?

At the bottom of:

- encoder
    
- decoder
    

The positional encoding is added to the token embeddings.

So:

Input=Embedding+PositionalEncodingInput = Embedding + PositionalEncoding

The dimensions must match:

512+512512+512

does **not** mean the result has 1024 dimensions.

They add the vectors element-by-element:

```
Embedding:
[0.2, 0.5, 0.1]

Position:
[0.1, 0.3, 0.8]

Add:
[0.3, 0.8, 0.9]
```

The resulting vector remains the same size.

---

# 64. The complete encoder

Let's put everything together.

Input:

```
"I love cats"
```

### Step 1: Tokenization

```
I
love
cats
```

### Step 2: Embedding

```
I     → vector
love  → vector
cats  → vector
```

### Step 3: Positional encoding

```
I     + position 1
love  + position 2
cats  + position 3
```

### Step 4: Encoder layer 1

```
Multi-head self-attention
        ↓
Residual + LayerNorm
        ↓
Feed-forward network
        ↓
Residual + LayerNorm
```

### Step 5

Repeat.

### Step 6

Repeat six times.

Output:

```
rich contextual representations
```

---

# 65. What does "contextual representation" mean?

This is extremely important.

The word:

> bank

can mean:

> financial institution

or:

> side of a river

An embedding that represents only the word itself may not know which meaning is intended.

After self-attention, the representation of `bank` can incorporate surrounding words.

For:

> "I deposited money in the bank."

the representation of `bank` can incorporate:

```
deposited
money
```

For:

> "We sat beside the river bank."

it can incorporate:

```
river
```

Thus the representation becomes:

> meaning-in-context

rather than simply:

> meaning-of-word.

---

# 66. Encoder-decoder attention

Now suppose we're translating:

> "I love cats."

The encoder has processed the English.

The decoder is trying to generate French.

At some point, the decoder might be generating:

> "chats"

It needs information from the English input.

So it uses:

```
Decoder representation
        ↓
      Query
        ↓
   attention
        ↑
Encoder representations
   Keys + Values
```

This is called:

# Encoder-decoder attention

The paper says every decoder position can attend to all positions in the input sequence through this mechanism. arXiv

---

# 67. Translation example

Input:

> "I love cats."

Encoder:

```
I
love
cats
```

produces contextual representations.

Decoder starts:

```
<START>
```

Then predicts:

```
J'
```

Then:

```
J'aime
```

Then:

```
J'aime les
```

Then:

```
J'aime les chats
```

Then:

```
<EOS>
```

During each step, the decoder can look at encoder representations.

---

# 68. The final output layer

The decoder produces a vector.

But we don't want a vector.

We want a word/token.

Suppose vocabulary size is:

37,00037,000

The model produces approximately:

```
37,000 scores
```

One for every possible token.

Then softmax converts those scores into probabilities.

For example:

```
cat       0.45
dog       0.12
house     0.03
the       0.20
...
```

Then the model selects/generates the next token according to the decoding strategy.

---

# 69. What is the output softmax?

Suppose the decoder outputs:

z=[2.1,0.3,1.4,...]z=[2.1,0.3,1.4,...]

The softmax turns these into probabilities.

So the final stage is approximately:

```
Decoder representation
        ↓
Linear layer
        ↓
scores for vocabulary
        ↓
softmax
        ↓
probability of each token
```

The paper explicitly describes using a learned linear transformation followed by softmax to obtain next-token probabilities. arXiv

---

# 70. Weight sharing

The paper shares the same weight matrix between:

- the two embedding layers
    
- the pre-softmax linear transformation
    

This is called:

# Weight tying / weight sharing

It reduces the number of independent parameters and connects the input/output representations.

The paper describes this choice in Section 3.4. arXiv

---

# 71. What is the Transformer "base" model?

The main base configuration is:

|Component|Value|
|---|---|
|Encoder layers|6|
|Decoder layers|6|
|dmodeld_{model}|512|
|dffd_{ff}|2048|
|Attention heads|8|
|dkd_k|64|
|dvd_v|64|
|Dropout|0.1|
|Label smoothing|0.1|

These values are given in the paper's model-variation table. arXiv

---

# 72. What is the Transformer "big" model?

The paper also evaluates a larger configuration:

```
6 encoder layers
6 decoder layers
dmodel = 1024
dff = 4096
16 attention heads
300K training steps
```

The paper reports 213 million parameters for this configuration. arXiv

---

# 73. Why is self-attention faster?

Now we return to one of the paper's main arguments.

For a sequence of length nn and representation dimension dd, self-attention has approximately:

O(n2d)O(n^2d)

complexity per layer.

Why n2n^2?

Because every position can compare itself against every other position.

If there are:

```
10 tokens
```

you have roughly:

```
10 × 10 = 100
```

relationships.

If:

```
1000 tokens
```

then:

```
1000 × 1000 = 1,000,000
```

relationships.

So self-attention becomes expensive for extremely long sequences.

This is one of the important limitations that later Transformer research works to address.

---

# 74. RNN complexity

The paper gives recurrent layers approximately:

O(nd2)O(nd^2)

and:

O(n)O(n)

sequential operations.

Self-attention has:

O(n2d)O(n^2d)

but:

O(1)O(1)

sequential operations.

This distinction is extremely important. arXiv+1

---

# 75. What does O(1)O(1) sequential operations mean?

It does **not** mean the Transformer takes one mathematical operation.

It means:

> The number of sequential steps doesn't grow with sequence length in the same way as an RNN.

The attention computations can be performed largely in parallel.

For example:

```
RNN:

word1 → word2 → word3 → word4 → word5


Transformer:

word1 ─┐
word2 ─┤
word3 ─┼──> parallel attention computation
word4 ─┤
word5 ─┘
```

---

# 76. Long-range path length

The paper also introduces an important concept:

# Path length

Suppose word A needs information from word B.

In an RNN:

```
A → intermediate → intermediate → intermediate → B
```

The path can be long.

In self-attention:

```
A ───────────── B
```

They can interact directly in one attention layer.

Therefore the maximum path length is:

### Self-attention

O(1)O(1)

### Recurrent

O(n)O(n)

### Convolution

Depending on architecture:

O(log⁡kn)O(\log_k n)

for dilated convolution in the paper's comparison.

These comparisons appear in Table 1. arXiv

---

# 77. What about convolutional models?

Before Transformer, another approach was:

# CNN / convolutional sequence models

A convolution looks at a local neighborhood.

For example:

```
word1 word2 word3
       ↑
     window
```

A single convolution doesn't necessarily connect every pair of positions.

You need multiple layers to pass information across the sequence.

The paper discusses architectures such as ConvS2S and ByteNet in this context. arXiv+1

---

# 78. Restricted self-attention

The authors acknowledge a problem:

Self-attention has:

O(n2d)O(n^2d)

cost.

For very long sequences, that's expensive.

They suggest restricting attention to a neighborhood of size rr.

Instead of:

```
every token ↔ every token
```

we could use:

```
token ↔ nearby tokens
```

Then complexity can become approximately:

O(rnd)O(rnd)

The tradeoff is that long-range paths become longer, approximately:

O(n/r)O(n/r)

The paper mentions this as a possible approach for very long sequences. arXiv

This idea eventually became very important in later efficient-attention research.

---

# 79. Training data

Now let's understand how they actually trained the model.

For English → German:

- approximately **4.5 million sentence pairs**
    
- shared vocabulary of approximately **37,000 tokens**
    
- using byte-pair encoding
    

For English → French:

- approximately **36 million sentences**
    
- approximately **32,000 word-piece vocabulary**
    

The paper describes batching by approximate sequence length and approximately 25,000 source tokens plus 25,000 target tokens per batch. arXiv

---

# 80. Why batch sentences by length?

Imagine:

```
Sentence A: 5 tokens
Sentence B: 6 tokens
Sentence C: 100 tokens
```

If you put these into a matrix, you have to pad shorter sentences.

Padding wastes computation.

Grouping similar-length sentences reduces wasted work.

---

# 81. What is padding?

Suppose:

```
I love cats
```

has 3 tokens.

Another sentence has 5:

```
I really love cute cats
```

To put them in the same batch:

```
I love cats <PAD> <PAD>
I really love cute cats
```

`<PAD>` is a special padding token.

The model usually masks padding so it doesn't treat padding as meaningful content.

---

# 82. Hardware

The researchers trained the models on:

# 8 NVIDIA P100 GPUs

The base model took approximately:

> 0.4 seconds per training step

and approximately:

> 100,000 steps / 12 hours

The larger model took approximately:

> 1 second per step

and:

> 300,000 steps / 3.5 days

according to the paper. arXiv

This was a major selling point of the architecture:

> strong performance while being much more parallelizable than recurrent alternatives.

---

# 83. The optimizer: Adam

The Transformer was trained with:

# Adam

Adam is an optimization algorithm.

Remember that training means:

```
make prediction
     ↓
calculate loss
     ↓
calculate gradients
     ↓
change parameters
```

Adam decides how to change parameters.

The paper uses:

β1=0.9\beta_1=0.9

β2=0.98\beta_2=0.98

ϵ=10−9\epsilon=10^{-9}

arXiv

---

# 84. What is a learning rate?

Suppose the model realizes:

> "My weights are wrong."

It needs to change them.

But how much?

That is controlled by:

# Learning rate

Too large:

```
overshoot
→ unstable
```

Too small:

```
learning extremely slowly
```

So choosing a good learning-rate schedule is important.

---

# 85. The famous Transformer learning-rate schedule

The paper uses:

lrate=dmodel−0.5⋅min⁡(step−0.5,step⋅warmup_steps−1.5)lrate = d_{model}^{-0.5} \cdot \min ( step^{-0.5}, step\cdot warmup\_steps^{-1.5} )

arXiv

This is often called the:

# Noam learning-rate schedule

although the paper itself simply gives the formula.

---

# 86. What is warmup?

At the beginning of training, instead of immediately using the full learning rate, they gradually increase it.

They use:

warmup_steps=4000warmup\_steps=4000

So approximately:

```
learning rate
      /\
     /  \
    /    \
   /      \____
  /
 /
--------------------> training
 0    4000
```

The paper says the learning rate increases during the first 4000 steps and then decreases proportionally to the inverse square root of the step number. arXiv

---

# 87. Why warmup?

Early in training, the network's parameters are not yet organized.

Large updates can destabilize learning.

Warmup lets training start gently.

Then the learning rate decreases as training progresses.

---

# 88. Regularization

Neural networks can:

# Overfit

That means they become too good at remembering the training data but worse at unseen examples.

The paper uses regularization techniques.

Specifically:

1. Dropout
    
2. Label smoothing
    

And residual dropout is applied at several locations. arXiv

---

# 89. What is dropout?

Imagine the neural network has:

```
● ● ● ● ● ● ● ●
```

During training, dropout randomly turns some connections/activations off:

```
● × ● ● × ● × ●
```

This prevents the network from depending too heavily on particular pathways.

The base model uses:

Pdrop=0.1P_{drop}=0.1

The paper applies dropout to sublayer outputs and to the sum of embeddings and positional encodings. arXiv

---

# 90. What is label smoothing?

Suppose the correct answer is:

```
cat = 1.0
dog = 0.0
horse = 0.0
```

A model trained this way can become excessively confident.

Label smoothing says, roughly:

> Don't force the model to assign exactly 100% probability to the correct class.

Instead of:

```
cat   = 1.0
dog   = 0.0
horse = 0.0
```

we might use something closer to:

```
cat   = 0.9
dog   = 0.05
horse = 0.05
```

The exact implementation distributes a small amount of probability mass according to the smoothing scheme.

The paper uses:

ϵls=0.1\epsilon_{ls}=0.1

and notes that this worsens perplexity while improving accuracy and BLEU. arXiv

---

# 91. What is perplexity?

Perplexity is a common language-model evaluation measure.

Very loosely:

> It measures how surprised the model is by the correct sequence.

Lower is generally better.

If a model confidently predicts the correct token:

```
"I am going to eat ___"

food = high probability
```

then it has low surprise.

If it predicts nonsense:

```
purple = 0.001
```

then it is very surprised when `food` is correct.

The paper reports per-wordpiece perplexities in its model-variation table and explicitly warns that these should not be compared directly with per-word perplexities because their tokenization uses BPE. arXiv

---

# 92. What is BLEU?

The paper primarily evaluates translation using:

# BLEU

BLEU is an automatic metric for comparing machine translations with reference translations.

It roughly checks how much the generated translation overlaps with human reference translations using n-gram precision, along with a penalty related to translation length.

Higher BLEU generally indicates closer correspondence to reference translations.

The paper's central translation results are reported using BLEU. arXiv+1

---

# 93. Main translation results

The paper reports for WMT 2014 English-German:

```
Transformer big:
28.4 BLEU
```

and reports strong results compared with the previous systems in the table.

For English-French, the paper's abstract reports:

```
41.8 BLEU
```

while the detailed Section 6.1 text in the PDF reports **41.0** for the big model. This discrepancy is present in the current arXiv PDF text itself, so it is worth being aware of rather than silently choosing one number. arXiv+1

---

# 94. What does "ensemble" mean?

An ensemble combines multiple models.

For example:

```
Model A → translation probabilities
Model B → translation probabilities
Model C → translation probabilities
             ↓
        combine them
```

Ensembles often improve performance because different models make different mistakes.

The Transformer paper is notable because the reported big Transformer achieved strong results even against earlier systems involving ensembles. arXiv

---

# 95. Why was this result surprising?

The authors had removed something people considered fundamental to sequence modeling:

# Recurrence

They also removed:

# Convolution

And instead relied almost entirely on:

# Attention

The paper describes the Transformer as a sequence transduction model based entirely on attention, using multi-head self-attention in place of recurrent layers. arXiv

That was the major conceptual breakthrough.

---

# 96. Model variations

The authors didn't simply build one model and stop.

They changed different components to determine what mattered.

This is important scientifically because it helps answer:

> "Which parts of the architecture are actually useful?"

Their experiments are summarized in Table 3. arXiv

---

# 97. Experiment A: Number of attention heads

The base model has:

88

heads.

They tested different numbers while keeping computation roughly controlled.

The table includes configurations such as:

```
1 head
4 heads
8 heads
16 heads
32 heads
```

The authors report that:

- single-head attention was 0.9 BLEU worse than the best setting
    
- too many heads also reduced quality
    

So:

> More heads isn't automatically better.

arXiv

---

# 98. Why could too many heads be bad?

Suppose you have eight heads with enough dimensions:

```
64 dimensions/head
```

But if you split the same total representation across 32 heads:

```
16 dimensions/head
```

Each head has much less representational capacity.

So there's a tradeoff:

```
too few heads
    ↓
not enough different relationships

too many heads
    ↓
each head becomes too small
```

---

# 99. Experiment B: Key dimension

They also changed:

dkd_k

The authors observed that reducing the key size hurt model quality.

They interpret this as suggesting that determining compatibility between query and key representations may be difficult, and that a more sophisticated compatibility function could potentially help. arXiv

---

# 100. Experiment C: Model size

They changed dimensions such as:

dmodeld_{model}

and:

dffd_{ff}

The larger models generally performed better.

This isn't surprising:

> More parameters can give the model more capacity to learn patterns.

But more parameters also means:

- more computation
    
- more memory
    
- more training cost
    

The paper's table shows this tradeoff explicitly. arXiv

---

# 101. Experiment D: Dropout

They tested different dropout rates.

The authors report that dropout was helpful for avoiding overfitting.

Again, this is an important experimental result:

> Regularization wasn't just included randomly; they tested its effect.

arXiv

---

# 102. Experiment E: Learned positional embeddings

They replaced sinusoidal positional encodings with learned positional embeddings.

The results were almost identical to the base model.

This means:

> The Transformer did not appear to depend specifically on sinusoidal encoding.

The authors chose sinusoidal encoding because of its possible extrapolation advantage. arXiv+1

---

# 103. English constituency parsing

The authors wanted to know:

> Is Transformer useful only for translation?

So they tested it on another task:

# English constituency parsing

This is a way of analyzing sentence structure.

For example:

> "The cat eats fish."

can be represented as a tree:

```
Sentence
├── Noun Phrase
│   └── The cat
└── Verb Phrase
    ├── eats
    └── fish
```

The goal is to identify this grammatical structure.

---

# 104. Why is parsing a difficult test?

The authors point out several challenges:

1. The output has strong structural constraints.
    
2. The output can be significantly longer than the input.
    
3. Earlier RNN sequence-to-sequence models struggled to reach state-of-the-art performance in small-data settings.
    

---

# 90. What is label smoothing?

Suppose the correct answer is:

```
cat = 1.0
dog = 0.0
horse = 0.0
```

> Don't force the model to assign exactly 100% probability to the correct class.

Instead of:

```
cat   = 1.0
dog   = 0.0
horse = 0.0
```

```
cat   = 0.9
dog   = 0.05
horse = 0.05
```

The exact implementation distributes a small amount of probability mass according to the smoothing scheme.

ϵls=0.1\epsilon_{ls}=0.1

---

# 91. What is perplexity?

Very loosely:

> It measures how surprised the model is by the correct sequence.

```
"I am going to eat ___"

food = high probability
```

then it has low surprise.

If it predicts nonsense:

```
purple = 0.001
```

then it is very surprised when `food` is correct.

The paper reports per-wordpiece perplexities in its model-variation table and explicitly warns that these should not be compared directly with per-word perplexities because their tokenization uses BPE. arXiv

---

# 92. What is BLEU?

The paper primarily evaluates translation using:

# BLEU

BLEU is an automatic metric for comparing machine translations with reference translations.

It roughly checks how much the generated translation overlaps with human reference translations using n-gram precision, along with a penalty related to translation length.

Higher BLEU generally indicates closer correspondence to reference translations.

The paper's central translation results are reported using BLEU. arXiv+1

---

# 93. Main translation results

The paper reports for WMT 2014 English-German:

```
Transformer big:
28.4 BLEU
```

and reports strong results compared with the previous systems in the table.

For English-French, the paper's abstract reports:

```
41.8 BLEU
```

while the detailed Section 6.1 text in the PDF reports **41.0** for the big model. This discrepancy is present in the current arXiv PDF text itself, so it is worth being aware of rather than silently choosing one number. arXiv+1

---

# 94. What does "ensemble" mean?

An ensemble combines multiple models.

For example:

```
Model A → translation probabilities
Model B → translation probabilities
Model C → translation probabilities
             ↓
        combine them
```

Ensembles often improve performance because different models make different mistakes.

The Transformer paper is notable because the reported big Transformer achieved strong results even against earlier systems involving ensembles. arXiv

---

# 95. Why was this result surprising?

The authors had removed something people considered fundamental to sequence modeling:

# Recurrence

They also removed:

# Convolution

And instead relied almost entirely on:

# Attention

The paper describes the Transformer as a sequence transduction model based entirely on attention, using multi-head self-attention in place of recurrent layers. arXiv

That was the major conceptual breakthrough.

---

# 96. Model variations

The authors didn't simply build one model and stop.

They changed different components to determine what mattered.

This is important scientifically because it helps answer:

> "Which parts of the architecture are actually useful?"

Their experiments are summarized in Table 3. arXiv

---

# 97. Experiment A: Number of attention heads

The base model has:

88

heads.

They tested different numbers while keeping computation roughly controlled.

The table includes configurations such as:

```
1 head
4 heads
8 heads
16 heads
32 heads
```

The authors report that:

- single-head attention was 0.9 BLEU worse than the best setting
    
- too many heads also reduced quality
    

> More heads isn't automatically better.

Suppose you have eight heads with enough dimensions:

```
64 dimensions/head
```

But if you split the same total representation across 32 heads:

```
16 dimensions/head
```

So there's a tradeoff:

```
too few heads
    ↓
not enough different relationships

too many heads
    ↓
each head becomes too small
```

---

# 99. Experiment B: Key dimension

They also changed:

dkd_k

The authors observed that reducing the key size hurt model quality.

They interpret this as suggesting that determining compatibility between query and key representations may be difficult, and that a more sophisticated compatibility function could potentially help.

---

# 100. Experiment C: Model size

They changed dimensions such as:

dmodeld_{model}

and:

dffd_{ff}

The larger models generally performed better.

This isn't surprising:

> More parameters can give the model more capacity to learn patterns.

But more parameters also means:

arXiv

---

# 105. Parsing dataset

They used the:

# Wall Street Journal portion of the Penn Treebank

For the WSJ-only experiment:

- approximately 40,000 training sentences
    
- vocabulary of 16,000 tokens
    

They also used a semi-supervised setup involving approximately 17 million sentences and a 32,000-token vocabulary. arXiv

---

# 106. Parsing results

The paper reports:

```
Transformer, WSJ only:
91.3 F1

Transformer, semi-supervised:
92.7 F1
```

on the specified WSJ evaluation set.

These results were competitive with or better than many earlier approaches listed in their table, although other systems in the table achieved higher scores in some settings. arXiv

The important conclusion isn't simply a ranking.

It's:

> **The architecture could generalize beyond machine translation.**

---

# 107. What is F1?

F1 combines:

- precision
    
- recall
    

into a single metric.

Very roughly:

### Precision

> Of the things I predicted, how many were correct?

### Recall

> Of the things that were actually correct, how many did I find?

F1 balances the two.

---

# 108. Beam search

The Transformer does not simply always choose the single most probable next token.

Instead, during translation, the paper uses:

# Beam search

with:

beam size=4beam\ size=4

for the main translation experiments.

The idea is:

```
Start
 ├── A
 ├── B
 ├── C
 └── D
```

Keep several promising partial translations.

Then expand them:

```
A → A1
  → A2

B → B1
  → B2

...
```

Keep the best few.

This allows the decoder to consider multiple possible sequences instead of committing too early to one token.

The paper also uses a length penalty. arXiv

---

# 109. What is length penalty?

Suppose the model prefers short outputs because multiplying probabilities can make longer sequences look less probable.

Then it might produce:

> "I love."

instead of:

> "I love cats very much."

A length penalty helps compensate for this bias.

The paper uses:

α=0.6\alpha=0.6

for its main translation beam search setup. arXiv

---

# 110. Checkpoint averaging

The researchers also averaged several model checkpoints.

For base models:

> last 5 checkpoints

For big models:

> last 20 checkpoints

The checkpoints were saved periodically.

This can smooth out fluctuations and produce a stronger final model. arXiv

---

# 111. Attention visualizations

The appendix contains visualizations of attention heads.

This is particularly interesting.

The researchers wanted to see:

> "What are the attention heads actually doing?"

They found examples where different heads seemed to learn different linguistic relationships.

The paper says many attention heads appeared related to syntactic and semantic structure. arXiv

---

# 112. Figure 3: Long-distance dependency

One visualization involves:

> "making ... more difficult"

The paper shows attention from the word:

> "making"

toward a distant dependency.

The authors note that many heads attend to a distant relationship that completes the phrase. arXiv

This demonstrates why self-attention is powerful.

Instead of passing information step-by-step through many intermediate words:

```
making
 ↓
...
 ↓
...
 ↓
more
 ↓
difficult
```

attention can directly connect relevant positions.

---

# 113. Figure 4: Anaphora resolution

The paper also examines:

> "The Law will never be perfect, but its application should be just..."

The word:

> "its"

needs to relate to:

> "The Law"

The paper shows two attention heads whose attention patterns appear involved in this relationship. arXiv

This is an example of:

# Anaphora resolution

Anaphora resolution means figuring out what a referring expression refers to.

For example:

> John went home because **he** was tired.

Who is "he"?

> John.

---

# 114. Figure 5

The final attention visualizations show that different heads appear to learn different structural behaviors.

The paper explicitly says that the heads clearly learned different tasks. arXiv

This is one of the fascinating properties of multi-head attention.

The authors didn't manually assign:

```
Head 1 = grammar
Head 2 = pronouns
Head 3 = verbs
```

Instead, these patterns emerged during training.

---

# 115. Why is the Transformer more parallelizable?

Let's compare.

## RNN

```
h1
 ↓
h2
 ↓
h3
 ↓
h4
 ↓
h5
```

You need the previous hidden state.

Therefore training has sequential dependencies.

## Transformer encoder

```
x1 ─┐
x2 ─┤
x3 ─┤
x4 ─┤──> matrix operations
x5 ─┤
x6 ─┘
```

The attention calculations can be performed using large matrix operations.

GPUs are extremely good at this.

This is one of the core reasons the architecture could train much faster than recurrent systems. arXiv+1

---

# 116. But wait: isn't Transformer still sequential?

This is an important subtlety.

## During training

The decoder's target sequence can be processed in parallel because the correct previous tokens are already known.

The causal mask prevents each position from looking into the future.

For example:

```
Input:

<START> I love cats

Position 1 → sees START
Position 2 → sees START, I
Position 3 → sees START, I, love
Position 4 → sees START, I, love, cats
```

All these masked attention computations can be calculated together.

## During generation

The model doesn't know the future tokens.

So generation remains autoregressive:

```
token 1
  ↓
token 2
  ↓
token 3
  ↓
token 4
```

The original paper explicitly identifies reducing generation's sequential nature as a future research goal. arXiv

---

# 117. This distinction is VERY important

The Transformer paper did **not** magically make text generation completely parallel.

It made the **training computation much more parallelizable**.

Generation still proceeds step by step in the original encoder-decoder Transformer.

That distinction is often misunderstood.

---

# 118. What does "Attention Is All You Need" really mean?

The title is deliberately provocative.

They aren't literally saying:

> "No other neural-network operations exist."

There are still:

- linear layers
    
- feed-forward networks
    
- normalization
    
- embeddings
    
- softmax
    
- residual connections
    
- positional encodings
    

This means:

> The Transformer did not appear to depend specifically on sinusoidal encoding.

The authors chose sinusoidal encoding because of its possible extrapolation advantage.

---

# 103. English constituency parsing

The authors wanted to know:

> Is Transformer useful only for translation?

> "The cat eats fish."

---

# 104. Why is parsing a difficult test?

The authors point out several challenges:

1. The output has strong structural constraints.
    
2. The output can be significantly longer than the input.
    
3. Earlier RNN sequence-to-sequence models struggled to reach state-of-the-art performance in small-data settings.
    

---

# 105. Parsing dataset

They used the:

# Wall Street Journal portion of the Penn Treebank

For the WSJ-only experiment:

- approximately 40,000 training sentences
    
- vocabulary of 16,000 tokens
    

They also used a semi-supervised setup involving approximately 17 million sentences and a 32,000-token vocabulary.

---

# 106. Parsing results

The title means:

> **You don't need recurrence or convolution as the main mechanism for modeling sequence relationships. Attention alone can replace them.**

That is the revolutionary claim.

---

# 119. The entire Transformer in one diagram

Here's the mental model I want you to remember:

```
                  INPUT TOKENS
                       │
                       ▼
                 TOKEN EMBEDDING
                       │
                       +
                       │
              POSITIONAL ENCODING
                       │
                       ▼
          ┌────────────────────────┐
          │      ENCODER × 6       │
          │                        │
          │ Multi-Head Self-Attn   │
          │          ↓             │
          │ Residual + LayerNorm   │
          │          ↓             │
          │ Feed Forward           │
          │          ↓             │
          │ Residual + LayerNorm   │
          └───────────┬────────────┘
                      │
                      │ encoder output
                      ▼
          ┌────────────────────────┐
          │       DECODER × 6      │
          │                        │
          │ Masked Self-Attention  │
          │          ↓             │
          │ Residual + LayerNorm   │
          │          ↓             │
          │ Encoder-Decoder Attn   │◄──── Encoder output
          │          ↓             │
          │ Residual + LayerNorm   │
          │          ↓             │
          │ Feed Forward           │
          │          ↓             │
          │ Residual + LayerNorm   │
          └───────────┬────────────┘
                      │
                      ▼
                 Linear Layer
                      │
                      ▼
                   Softmax
                      │
                      ▼
              NEXT TOKEN PROBABILITY
```

That is the original Transformer.

---

# 120. Let's follow one word through the entire model

Suppose the input is:

> "I love cats."

Let's follow `cats`.

### Step 1 — Tokenization

```
cats → token ID
```

### Step 2 — Embedding

```
cats → 512-dimensional vector
```

### Step 3 — Position

Suppose `cats` is position 3.

Add its positional encoding.

```
cats embedding
       +
position 3 encoding
       ↓
combined representation
```

### Step 4 — Self-attention

`cats` looks at:

```
I
love
cats
.
```

It calculates attention weights.

### Step 5 — Feed-forward

The resulting representation goes through:

```
512 → 2048 → 512
```

### Step 6

This happens again in encoder layer 2.

### Step 7

Again in layer 3.

### Step 8

Again in layer 4.

### Step 9

Again in layer 5.

### Step 10

Again in layer 6.

Now `cats` has a highly processed contextual representation.

---

# 121. Then the decoder

Suppose we're translating into French.

The decoder may have generated:

```
J'aime les
```

Now it needs the next word.

The decoder:

1. Looks at previously generated French tokens using masked self-attention.
    
2. Looks at the English encoder representations using encoder-decoder attention.
    
3. Processes the result through its feed-forward network.
    
4. Produces vocabulary scores.
    
5. Applies softmax.
    
6. Predicts something like:
    

```
chats
```

Then it repeats.

---

# 122. A crucial distinction: self-attention vs cross-attention

You should memorize this distinction.

## Self-attention

Q, K, V come from the same sequence.

```
Sequence
   ↓
Q K V
   ↓
attention
```

Used in:

- encoder
    
- decoder
    

## Cross-attention / encoder-decoder attention

Q comes from decoder.

K and V come from encoder.

```
Decoder → Q

Encoder → K,V
```

Used in the decoder.

---

# 123. Another crucial distinction: attention vs multi-head attention

## Attention

One attention operation:

softmax(QKTdk)Vsoftmax \left( \frac{QK^T}{\sqrt{d_k}} \right)V

## Multi-head attention

Several attention operations in parallel:

headi=Attention(QWiQ,KWiK,VWiV)head_i = Attention(QW_i^Q,KW_i^K,VW_i^V)

then concatenate them.

So:

```
Attention
    ↓
one attention mechanism

Multi-head attention
    ↓
many attention mechanisms operating in parallel
```

---

# 124. Another crucial distinction: embedding vs positional encoding

## Embedding

Answers:

> What does this token represent?

## Positional encoding

Answers:

> Where is this token in the sequence?

Combined:

```
"What is this?"
+
"Where is it?"
```

---

# 125. Another crucial distinction: encoder vs decoder

## Encoder

Primarily:

> Understand the input.

## Decoder

Primarily:

> Generate the output.

And the decoder gets information from the encoder through cross-attention.

---

# 126. Another crucial distinction: training vs inference

## Training

We know the target sequence.

So we can process many target positions in parallel using masking.

## Inference

We don't know the target.

So:

```
predict token
     ↓
feed it back
     ↓
predict next
     ↓
feed it back
     ↓
...
```

This is why generation is sequential.

---

# 127. What is the paper's central contribution?

There are several.

### Contribution 1

A sequence-to-sequence architecture based entirely on attention rather than recurrence or convolution.

### Contribution 2

Multi-head self-attention as the primary mechanism.

### Contribution 3

Highly parallelizable training.

### Contribution 4

Strong translation performance.

### Contribution 5

Demonstration that the architecture generalizes to another task, constituency parsing.

The authors summarize the main contribution as replacing recurrent layers with multi-headed self-attention. arXiv

---

# 128. Why did this paper become so historically important?

Because it changed the default way researchers thought about sequence modeling.

Before:

```
RNN
LSTM
GRU
CNN
```

were major sequence-modeling tools.

After Transformer:

```
Attention
   ↓
Transformer
   ↓
BERT
GPT
T5
ViT
modern multimodal models
...
```

Many later architectures are descendants, modifications, or conceptual extensions of Transformer ideas.

The paper itself, however, is much simpler than today's huge language models.

It was originally designed and evaluated primarily for translation.

---

# 129. What the paper did NOT introduce

This is also important.

The paper did **not** introduce:

- modern GPT-style decoder-only language models
    
- instruction tuning
    
- RLHF
    
- chatbots
    
- huge-scale pretraining
    
- billions/trillions of parameters
    
- modern tokenizers such as today's common BPE variants in exactly their current implementations
    
- KV caching
    
- FlashAttention
    
- mixture-of-experts Transformers
    
- RoPE
    
- ALiBi
    
- modern long-context attention mechanisms
    

The Transformer does not simply always choose the single most probable next token.

Instead, during translation, the paper uses:

# Beam search

beam size=4beam\ size=4

for the main translation experiments.

Then expand them:

```
A → A1
  → A2

B → B1
  → B2

...
```

Keep the best few.

This allows the decoder to consider multiple possible sequences instead of committing too early to one token.

The paper also uses a length penalty. arXiv

---

# 109. What is length penalty?

Suppose the model prefers short outputs because multiplying probabilities can make longer sequences look less probable.

Then it might produce:

> "I love."

instead of:

> "I love cats very much."

A length penalty helps compensate for this bias.

The paper uses:

α=0.6\alpha=0.6

for its main translation beam search setup. arXiv

---

# 110. Checkpoint averaging

The researchers also averaged several model checkpoints.

For base models:

> last 5 checkpoints

For big models:

> last 20 checkpoints

The checkpoints were saved periodically.

This can smooth out fluctuations and produce a stronger final model. arXiv

---

# 111. Attention visualizations

The appendix contains visualizations of attention heads.

This is particularly interesting.

The researchers wanted to see:

> "What are the attention heads actually doing?"

They found examples where different heads seemed to learn different linguistic relationships.

The paper says many attention heads appeared related to syntactic and semantic structure. arXiv

---

# 112. Figure 3: Long-distance dependency

One visualization involves:

> "making ... more difficult"

The paper shows attention from the word:

> "making"

toward a distant dependency.

The authors note that many heads attend to a distant relationship that completes the phrase. arXiv

This demonstrates why self-attention is powerful.

Instead of passing information step-by-step through many intermediate words:

```
making
 ↓
...
 ↓
...
 ↓
more
 ↓
difficult
```

attention can directly connect relevant positions.

---

# 113. Figure 4: Anaphora resolution

The paper also examines:

> "The Law will never be perfect, but its application should be just..."

The word:

> "its"

needs to relate to:

> "The Law"

The paper shows two attention heads whose attention patterns appear involved in this relationship. arXiv

This is an example of:

# Anaphora resolution

Anaphora resolution means figuring out what a referring expression refers to.

For example:

> John went home because **he** was tired.

Who is "he"?

> John.

---

# 114. Figure 5

The final attention visualizations show that different heads appear to learn different structural behaviors.

The paper explicitly says that the heads clearly learned different tasks. arXiv

This is one of the fascinating properties of multi-head attention.

The authors didn't manually assign:

```
Head 1 = grammar
Head 2 = pronouns
Head 3 = verbs
```

Instead, these patterns emerged during training.

---

# 115. Why is the Transformer more parallelizable?

Let's compare.

## RNN

```
h1
 ↓
h2
 ↓
h3
 ↓
h4
 ↓
h5
```

You need the previous hidden state.

Therefore training has sequential dependencies.

## Transformer encoder

```
x1 ─┐
x2 ─┤
x3 ─┤
x4 ─┤──> matrix operations
x5 ─┤
x6 ─┘
```

The attention calculations can be performed using large matrix operations.

GPUs are extremely good at this.

This is one of the core reasons the architecture could train much faster than recurrent systems. arXiv+1

---

# 116. But wait: isn't Transformer still sequential?

This is an important subtlety.

## During training

The decoder's target sequence can be processed in parallel because the correct previous tokens are already known.

The causal mask prevents each position from looking into the future.

For example:

```
Input:

<START> I love cats

Position 1 → sees START
Position 2 → sees START, I
Position 3 → sees START, I, love
Position 4 → sees START, I, love, cats
```

All these masked attention computations can be calculated together.

## During generation

The model doesn't know the future tokens.

So generation remains autoregressive:

```
token 1
  ↓
token 2
  ↓
token 3
  ↓
token 4
```

The original paper explicitly identifies reducing generation's sequential nature as a future research goal. arXiv

---

# 117. This distinction is VERY important

The Transformer paper did **not** magically make text generation completely parallel.

It made the **training computation much more parallelizable**.

Generation still proceeds step by step in the original encoder-decoder Transformer.

That distinction is often misunderstood.

---

# 118. What does "Attention Is All You Need" really mean?

The title is deliberately provocative.

They aren't literally saying:

> "No other neural-network operations exist."

There are still:

- linear layers
    
- feed-forward networks
    
- normalization
    
- embeddings
    
- softmax
    
- residual connections
    
- positional encodings
    

```
                  INPUT TOKENS
                       │
                       ▼
                 TOKEN EMBEDDING
                       │
                       +
                       │
              POSITIONAL ENCODING
                       │
                       ▼
          ┌────────────────────────┐
          │      ENCODER × 6       │
          │                        │
          │ Multi-Head Self-Attn   │
          │          ↓             │
          │ Residual + LayerNorm   │
          │          ↓             │
          │ Feed Forward           │
          │          ↓             │
          │ Residual + LayerNorm   │
          └───────────┬────────────┘
                      │
                      │ encoder output
                      ▼
          ┌────────────────────────┐
          │       DECODER × 6      │
          │                        │
          │ Masked Self-Attention  │
          │          ↓             │
          │ Residual + LayerNorm   │
          │          ↓             │
          │ Encoder-Decoder Attn   │◄──── Encoder output
          │          ↓             │
          │ Residual + LayerNorm   │
          │          ↓             │
          │ Feed Forward           │
          │          ↓             │
          │ Residual + LayerNorm   │
          └───────────┬────────────┘
                      │
                      ▼
                 Linear Layer
                      │
                      ▼
                   Softmax
                      │
                      ▼
              NEXT TOKEN PROBABILITY
```

---

> "I love cats."

Let's follow `cats`.

```
cats → token ID
```

### Step 2 — Embedding

```
cats → 512-dimensional vector
```

### Step 3 — Position

```
cats embedding
       +
position 3 encoding
       ↓
combined representation
```

### Step 4 — Self-attention

`cats` looks at:

```
I
love
cats
.
```

### Step 5 — Feed-forward

The resulting representation goes through:

```
512 → 2048 → 512
```

### Step 6

### Step 7

Again in layer 3.

### Step 8

### Step 9

Again in layer 5.

### Step 10

Now `cats` has a highly processed contextual representation.

Suppose we're translating into French.

The decoder may have generated:

```
J'aime les
```

Now it needs the next word.

The decoder:

1. Looks at previously generated French tokens using masked self-attention.
    
2. Looks at the English encoder representations using encoder-decoder attention.
    
3. Processes the result through its feed-forward network.
    
4. Produces vocabulary scores.
    
5. Applies softmax.
    
6. Predicts something like:
    

```
chats
```

---

# 122. A crucial distinction: self-attention vs cross-attention

## Self-attention

Q, K, V come from the same sequence.

```
Sequence
   ↓
Q K V
   ↓
attention
```

Used in:

- encoder
    
- decoder
    

## Cross-attention / encoder-decoder attention

Those came later.

The 2017 paper provides the foundational Transformer architecture.

---

# 130. Original Transformer vs modern GPT

The original Transformer:

```
Encoder
+
Decoder
```

Modern GPT-style models are generally:

```
Decoder-only Transformer
```

They remove the separate encoder and use causal self-attention.

So if you understand this paper, you have a very strong foundation for understanding GPT architecture.

But don't make the mistake:

> "GPT = exactly the original Transformer."

It isn't.

---

# 131. Why decoder-only models can work

A decoder-only model can simply learn:

P(xt∣x1,…,xt−1)P(x_t|x_1,\ldots,x_{t-1})

In plain English:

> Given everything before this point, what token should come next?

That's next-token prediction.

The original paper instead focuses on encoder-decoder sequence transformation such as translation.

---

# 132. The mathematical heart of the paper

If you remember only a handful of equations, remember these.

## Attention

Attention(Q,K,V)=softmax(QKTdk)VAttention(Q,K,V) = softmax \left( \frac{QK^T}{\sqrt{d_k}} \right)V

## Multi-head attention

MultiHead(Q,K,V)=Concat(head1,…,headh)WOMultiHead(Q,K,V) = Concat(head_1,\ldots,head_h)W^O

where:

headi=Attention(QWiQ,KWiK,VWiV)head_i= Attention(QW_i^Q,KW_i^K,VW_i^V)

## Feed-forward

FFN(x)=max(0,xW1+b1)W2+b2FFN(x) = max(0,xW_1+b_1)W_2+b_2

## Positional encoding

PE(pos,2i)=sin⁡(pos100002i/dmodel)PE(pos,2i) = \sin \left( \frac{pos}{10000^{2i/d_{model}}} \right)

PE(pos,2i+1)=cos⁡(pos100002i/dmodel)PE(pos,2i+1) = \cos \left( \frac{pos}{10000^{2i/d_{model}}} \right)

These equations represent much of the architecture's mathematical core. arXiv+2

---

# 133. If you only remember one equation

Remember:

Attention(Q,K,V)=softmax(QKTdk)V\boxed{ Attention(Q,K,V) = softmax \left( \frac{QK^T}{\sqrt{d_k}} \right)V }

In human language:

> **Compare queries with keys → scale the scores → turn them into weights → use those weights to combine the values.**

That's attention.

---

# 134. If you only remember one architecture

Remember:

```
Input
 ↓
Embedding + Position
 ↓
Self-Attention
 ↓
Feed Forward
 ↓
repeat
 ↓
Encoder representation
 ↓
Decoder
 ↓
Masked Self-Attention
 ↓
Cross-Attention
 ↓
Feed Forward
 ↓
repeat
 ↓
Linear
 ↓
Softmax
 ↓
Next token
```

---

# 135. What problem does each component solve?

This table is worth memorizing.

|Component|Problem it solves|
|---|---|
|Tokenization|Turns text into manageable pieces|
|Embedding|Turns tokens into useful numerical vectors|
|Positional encoding|Gives the model information about order|
|Self-attention|Lets tokens interact with other tokens|
|Scaling|Keeps dot-product scores numerically well-behaved|
|Softmax|Converts scores into attention weights|
|Multi-head attention|Lets model learn different relationships simultaneously|
|Feed-forward network|Performs nonlinear processing on each position|
|Residual connection|Helps information/gradients flow through deep networks|
|LayerNorm|Stabilizes representations|
|Masking|Prevents decoder from seeing future tokens|
|Cross-attention|Lets decoder access encoder information|
|Output softmax|Converts decoder scores into token probabilities|
|Dropout|Reduces overfitting|
|Label smoothing|Prevents excessive confidence|
|Adam|Optimizes model parameters|
|Learning-rate warmup|Makes early training more stable|
|Beam search|Searches multiple promising output sequences|

---

# 136. The paper's logic from beginning to end

Now let's compress the entire paper's argument.

### Problem

RNN-based sequence models are sequential.

↓

### Consequence

Training cannot be fully parallelized.

↓

### Existing solution

Attention helps RNNs model distant relationships.

↓

### New idea

Why not remove the RNN completely?

↓

### Transformer

Use attention as the primary sequence-processing mechanism.

↓

### Self-attention

Every position can directly interact with every other position.

↓

### Multi-head attention

Use multiple attention mechanisms to capture different relationships.

↓

### Positional encoding

Because there is no recurrence, explicitly provide position information.

↓

### Encoder-decoder architecture

Encoder understands the source.

Decoder generates the target.

↓

### Masking

Prevent decoder from seeing future outputs.

↓

### Result

Strong translation quality with much more parallelizable training.

↓

### Additional test

Apply Transformer to constituency parsing.

↓

### Result

It generalizes beyond translation.

That's the paper.

---

# 137. The deepest intuition

Imagine a classroom.

There are 30 students.

A traditional sequential system works like:

```
Student 1 talks to Student 2
Student 2 talks to Student 3
Student 3 talks to Student 4
...
```

Information takes time to travel.

Self-attention is more like:

> Everyone can simultaneously look at everyone else and decide whose information is relevant.

Multi-head attention means:

> We have multiple ways of looking at the classroom.

One person might pay attention to:

```
who is sitting nearby
```

another to:

```
who is discussing the same topic
```

another to:

```
who is answering the teacher's question
```

The Transformer learns these patterns automatically.

---

# 138. Why "attention" is such a good name

Suppose you're reading:

> "The dog chased the cat because it was hungry."

When interpreting:

> "it"

your brain doesn't give equal importance to every word.

You naturally focus more on relevant words.

Attention mechanisms mathematically approximate this kind of selective information retrieval.

The model isn't conscious and doesn't "understand" language in the human sense, but mathematically it learns patterns of relationships between representations.

---

# 139. What does the model actually learn?

The researchers don't manually program:

```
"it" refers to previous noun
```

or:

```
verbs should attend to subjects
```

Instead, training gives the model examples.

The model has millions of parameters.

Through gradient-based optimization, those parameters change so that the final predictions become better.

Over training, some attention heads can develop patterns that correlate with:

- syntactic relationships
    
- semantic relationships
    
- long-distance dependencies
    
- anaphora
    
- other linguistic structures
    

The attention visualizations in the paper provide examples of this behavior. arXiv+1

---

# 140. What does a Transformer layer actually "know"?

This is an important philosophical point.

A layer does not contain a little dictionary saying:

```
cat = animal
dog = animal
```

Instead, information is distributed across large vectors and learned transformations.

For example, the representation of:

```
bank
```

after several layers can encode information influenced by:

```
money
deposit
account
```

or:

```
river
water
shore
```

depending on context.

The representation is therefore **distributed**.

---

# 141. Why six layers?

The paper chooses:

N=6N=6

for both encoder and decoder in the base model.

This is an architectural hyperparameter.

The authors also test other numbers of layers in their model variations and find that larger models generally provide more capacity. arXiv

There is nothing magical about the number 6.

It's a design choice that worked well for their experiments.

---

# 142. Why 512 dimensions?

Again:

> Not a law of nature.

It's a hyperparameter.

The model's internal representation has 512 numbers in the base configuration.

The big model increases this to 1024.

The authors test different dimensions and find that model size affects performance. arXiv

Modern models can use vastly different dimensions.

---

# 143. Why 8 heads?

Also a hyperparameter.

The base model:

h=8h=8

The big model:

h=16h=16

The experiments demonstrate that both too few and too many heads can hurt performance under their fixed-computation comparisons. arXiv

---

# 144. Why dmodel=512d_{model}=512, dk=64d_k=64?

Because:

8×64=5128\times64=512

This lets eight attention heads together represent the original model dimension.

---

# 145. Why does attention need both K and V?

This is a subtle but important question.

The key is used for:

> deciding relevance.

The value is used for:

> providing information.

You can think of:

```
Key:
"What kind of information do I have?"

Value:
"Here is the actual information."
```

That separation gives attention flexibility.

The model can learn one representation for matching and another for information transfer.

---

# 146. Why isn't attention simply "weighted averaging"?

In one sense, it is.

The output is a weighted sum of value vectors.

But the important thing is:

> **The weights are dynamically calculated based on the current queries and keys.**

So it isn't a fixed average.

For one sentence:

```
animal → 0.7
road → 0.1
...
```

For another:

```
animal → 0.1
road → 0.7
...
```

The weights depend on context.

---

# 147. Why multiple attention heads matter

Imagine one person has only one camera.

They must capture everything with one viewpoint.

Now imagine eight cameras:

```
Camera 1 → grammar
Camera 2 → nearby words
Camera 3 → long-distance relation
Camera 4 → subject/object
...
```

The actual Transformer doesn't explicitly assign these roles, but different heads can learn different useful patterns.

That's the intuition behind multi-head attention.

---

# 148. Why does attention have n2n^2 complexity?

Suppose:

n=4n=4

tokens:

```
A B C D
```

Every token can attend to every token:

```
A → A B C D
B → A B C D
C → A B C D
D → A B C D
```

That's:

4×4=164\times4=16

relationships.

For nn tokens:

n2n^2

relationships.

This is why Transformers become expensive for very long sequences.

---

# 149. Why was this tradeoff acceptable?

The authors argue that for typical sentence representations at the time, sequence lengths were often smaller than the representation dimension, making self-attention computationally attractive compared with recurrent alternatives.

They also note that restricted attention could be used for very long sequences. arXiv

---

# 150. What the authors considered the biggest advantages

The paper emphasizes three major properties:

### 1. Computational complexity

How much work per layer?

### 2. Parallelization

How many sequential operations are required?

### 3. Path length

How many steps must information travel between distant positions?

Self-attention provides:

```
constant path length
high parallelism
```

at the cost of:

```
quadratic interaction cost in sequence length
```

This is the central engineering tradeoff. arXiv

---

# 151. The paper's conclusion

The authors conclude that they introduced a sequence transduction model based entirely on attention and replaced recurrent layers with multi-headed self-attention.

They report that the Transformer could be trained significantly faster than recurrent or convolutional architectures while achieving strong translation performance, and they demonstrate generalization to parsing. arXiv

They also identify future directions including:

- other input/output modalities
    
- images
    
- audio
    
- video
    
- restricted/local attention for large inputs and outputs
    
- reducing sequential generation
    

arXiv

---

# 152. One gigantic mental picture

If you remember nothing else, remember this:

```
                    TRANSFORMER
                         │
          ┌──────────────┴──────────────┐
          │                             │
       ENCODER                       DECODER
          │                             │
     understands                   generates
      the input                     output
          │                             │
          │                     masked self-attention
          │                             │
          │                     cross-attention
          │                             │
          │                       feed-forward
          │                             │
          └──────────────┐              │
                         │              │
                         └──────────────┘
                                │
                              output
                                │
                              softmax
                                │
                           next token
```

Inside the attention mechanism:

```
             QUERY
               │
               ▼
          compare with
               │
               ▼
             KEYS
               │
               ▼
          attention scores
               │
               ▼
             softmax
               │
               ▼
       attention weights
               │
               ▼
            VALUES
               │
               ▼
             OUTPUT
```

And multi-head attention does this multiple times:

```
                Input
                  │
       ┌──────────┼──────────┐
       ↓          ↓          ↓
     Head 1     Head 2     Head 3 ... Head 8
       │          │          │
       └──────────┼──────────┘
                  ↓
              Concatenate
                  ↓
             Linear layer
```

---

# 153. The complete paper in extremely simple language

If I had to explain the entire paper to a 10th-grade student in one story:

> Computers need to process language as numbers.
> 
> Older systems such as RNNs read language one piece at a time. This made it difficult to train them in parallel and made long-distance relationships harder to handle.
> 
> The researchers asked: **What if every word could directly look at every other word?**
> 
> They created **self-attention**.
> 
> Self-attention lets each token calculate which other tokens are relevant.
> 
> It does this using **queries, keys and values**.
> 
> Queries are compared with keys to calculate attention scores.
> 
> The scores are scaled and passed through softmax.
> 
> Those probabilities are used to combine the values.
> 
> Instead of using one attention operation, the Transformer uses several attention heads simultaneously. This is **multi-head attention**.
> 
> Because attention itself doesn't know word order, the researchers add **positional encodings**.
> 
> The encoder repeatedly applies self-attention and a feed-forward network.
> 
> The decoder does the same thing, but also looks at the encoder through **encoder-decoder attention**.
> 
> The decoder is prevented from seeing future output tokens using a **mask**.
> 
> Finally, the decoder produces probabilities for the next token using a linear layer and softmax.
> 
> The entire system is trained using gradient-based optimization with Adam, learning-rate warmup, dropout and label smoothing.
> 
> The result was a model that achieved excellent translation performance while being much more parallelizable than recurrent architectures.
> 
> The architecture also worked on constituency parsing.
> 
> This architecture became the foundation for a huge amount of subsequent deep-learning research.

---

# 154. The 20 words you absolutely need to know

If you're beginning from zero, learn these first:

```
Decoder-only Transformer
```

They remove the separate encoder and use causal self-attention.

But don't make the mistake:

> "GPT = exactly the original Transformer."

It isn't.

---

# 131. Why decoder-only models can work

A decoder-only model can simply learn:

P(xt∣x1,…,xt−1)P(x_t|x_1,\ldots,x_{t-1})

> Given everything before this point, what token should come next?

---

# 132. The mathematical heart of the paper

## Attention

## Multi-head attention

MultiHead(Q,K,V)=Concat(head1,…,headh)WOMultiHead(Q,K,V) = Concat(head_1,\ldots,head_h)W^O

## Feed-forward

PE(pos,2i+1)=cos⁡(pos100002i/dmodel)PE(pos,2i+1) = \cos \left( \frac{pos}{10000^{2i/d_{model}}} \right)

---

# 133. If you only remember one equation

```
Input
 ↓
Embedding + Position
 ↓
Self-Attention
 ↓
Feed Forward
 ↓
repeat
 ↓
Encoder representation
 ↓
Decoder
 ↓
Masked Self-Attention
 ↓
Cross-Attention
 ↓
Feed Forward
 ↓
repeat
 ↓
Linear
 ↓
Softmax
 ↓
Next token
```

---

# 135. What problem does each component solve?

This table is worth memorizing.

|Component|Problem it solves|
|---|---|
|Tokenization|Turns text into manageable pieces|
|Embedding|Turns tokens into useful numerical vectors|
|Positional encoding|Gives the model information about order|
|Self-attention|Lets tokens interact with other tokens|
|Scaling|Keeps dot-product scores numerically well-behaved|
|Softmax|Converts scores into attention weights|
|Multi-head attention|Lets model learn different relationships simultaneously|
|Feed-forward network|Performs nonlinear processing on each position|
|Residual connection|Helps information/gradients flow through deep networks|
|LayerNorm|Stabilizes representations|
|Masking|Prevents decoder from seeing future tokens|
|Cross-attention|Lets decoder access encoder information|
|Output softmax|Converts decoder scores into token probabilities|
|Dropout|Reduces overfitting|
|Label smoothing|Prevents excessive confidence|
|Adam|Optimizes model parameters|
|Learning-rate warmup|Makes early training more stable|
|Beam search|Searches multiple promising output sequences|

---

# 136. The paper's logic from beginning to end

### Problem

↓

1. **Token** — piece of text processed by the model.
    
2. **Embedding** — numerical vector representing a token.
    
3. **Vector** — list of numbers.
    
4. **Matrix** — table of numbers.
    
5. **Parameter** — number learned during training.
    
6. **Attention** — mechanism for deciding what information is relevant.
    
7. **Query** — what a token is looking for.
    
8. **Key** — representation used to determine relevance.
    
9. **Value** — information that gets passed through attention.
    
10. **Self-attention** — attention within the same sequence.
    
11. **Multi-head attention** — several attention mechanisms operating in parallel.
    
12. **Encoder** — processes/represents the input.
    
13. **Decoder** — generates the output.
    
14. **Mask** — prevents attention to forbidden positions.
    
15. **Positional encoding** — tells the model where tokens occur.
    
16. **Feed-forward network** — nonlinear processing applied to each position.
    
17. **Residual connection** — adds the original input back to a layer's output.
    
18. **LayerNorm** — normalizes activations.
    
19. **Softmax** — converts scores into probability-like values.
    
20. **Autoregressive** — generating the next token using previous tokens.
    

---

# 155. The five equations to master

If you eventually want to understand modern LLM papers, these five are a fantastic starting point.

### Equation 1 — Attention

Attention(Q,K,V)=softmax(QKTdk)V\boxed{ Attention(Q,K,V) = softmax \left( \frac{QK^T}{\sqrt{d_k}} \right)V }

### Equation 2 — Feed-forward network

FFN(x)=max(0,xW1+b1)W2+b2\boxed{ FFN(x) = max(0,xW_1+b_1)W_2+b_2 }

### Equation 3 — Positional encoding

PE(pos,2i)=sin⁡(pos100002i/dmodel)\boxed{ PE(pos,2i) = \sin \left( \frac{pos}{10000^{2i/d_{model}}} \right) }

### Equation 4 — Other positional dimension

PE(pos,2i+1)=cos⁡(pos100002i/dmodel)\boxed{ PE(pos,2i+1) = \cos \left( \frac{pos}{10000^{2i/d_{model}}} \right) }

### Equation 5 — Learning-rate schedule

lrate=dmodel−0.5min⁡(step−0.5,step⋅warmup_steps−1.5)\boxed{ lrate = d_{model}^{-0.5} \min ( step^{-0.5}, step\cdot warmup\_steps^{-1.5} ) }

All five come directly from the paper. arXiv+2

---

# 156. The single most important intuition

Finally, forget the equations for a moment.

Imagine every word in a sentence has a microphone.

A word can ask:

> **"Which other words should I listen to?"**

Attention answers that question.

Then multi-head attention says:

> **"Let's ask that question in several different ways."**

The encoder does this repeatedly to build rich representations.

The decoder does this repeatedly while generating the output.

That's the Transformer.

---

## What you should learn next

If your goal is to go from **absolute beginner → able to read Transformer/LLM papers**, the natural progression is:

```
1. Basic Python
       ↓
2. Linear algebra
   vectors, matrices, dot products
       ↓
3. Basic probability
   probabilities, distributions, softmax
       ↓
4. Neural networks
   neurons, layers, activation functions
       ↓
5. Gradient descent
       ↓
6. Backpropagation
       ↓
7. Word embeddings
       ↓
8. RNN / LSTM
       ↓
9. Attention
       ↓
10. Transformer
       ↓
11. BERT / GPT
       ↓
12. Modern LLM architecture
```

And **the most useful next step from this paper is not another paper**. It is to take the attention equation and calculate a tiny example by hand:

QKT→QKTdk→softmax→×VQK^T \rightarrow \frac{QK^T}{\sqrt{d_k}} \rightarrow softmax \rightarrow \times V

Once you can manually calculate that for, say, **3 tokens with 2-dimensional vectors**, the Transformer stops feeling like magic and starts looking like ordinary matrix multiplication.

The original paper is only 15 pages, but it packs an enormous amount of machinery into those pages. The explanation above covers its architecture, equations, motivation, computational comparison, training procedure, regularization, experiments, model variations, parsing experiment, attention visualizations, and conclusions. arXiv+5

If you want to truly **master** it rather than just understand the explanation, the next useful exercise is to implement a **tiny Transformer from scratch**, starting with just **`Q`, `K`, `V` and softmax**, and then gradually adding multi-head attention, positional encoding, residual connections, LayerNorm, encoder/decoder layers, masking, and finally training it.
