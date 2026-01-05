import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 1. Config
SRC_VOCAB = 100
TGT_VOCAB = 100
D_MODEL = 32
HEADS = 2
LAYERS = 2

# 2. Dummy Data
def get_data():
    src = tf.random.uniform((64, 5), minval=1, maxval=SRC_VOCAB, dtype=tf.int32)
    tgt = tf.reverse(src, axis=[1]) # Alien Language: Reverse
    return src, tgt

# 3. Model Definition (Keras Functional API)
def get_transformer():
    # Inputs
    src_input = layers.Input(shape=(5,), name="src")
    tgt_input = layers.Input(shape=(5,), name="tgt")
    
    # Embedding + Positional
    # Note: Keras usually handles Positional Encoding manually too, 
    # but for brevity we rely on the Dense layers to learn it implicitly or skip it for this toy task.
    src_emb = layers.Embedding(SRC_VOCAB, D_MODEL)(src_input)
    tgt_emb = layers.Embedding(TGT_VOCAB, D_MODEL)(tgt_input)
    
    # Encoder
    # Uses MultiHeadAttention
    attn_output = layers.MultiHeadAttention(num_heads=HEADS, key_dim=D_MODEL)(src_emb, src_emb)
    enc_output = layers.LayerNormalization()(src_emb + attn_output) # Add & Norm
    
    # Decoder
    # 1. Self Attention (Causal)
    attn_output_1 = layers.MultiHeadAttention(num_heads=HEADS, key_dim=D_MODEL)(tgt_emb, tgt_emb, use_causal_mask=True)
    dec_output_1 = layers.LayerNormalization()(tgt_emb + attn_output_1)
    
    # 2. Cross Attention (Query=Decoder, Value=Encoder)
    attn_output_2 = layers.MultiHeadAttention(num_heads=HEADS, key_dim=D_MODEL)(dec_output_1, enc_output)
    dec_output_2 = layers.LayerNormalization()(dec_output_1 + attn_output_2)
    
    # Final Output
    outputs = layers.Dense(TGT_VOCAB)(dec_output_2)
    
    return keras.Model(inputs=[src_input, tgt_input], outputs=outputs)

# 4. Training
model = get_transformer()
model.compile(
    optimizer="adam", 
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True)
)

print("🌊 Keras Training Start...")
x_src, x_tgt = get_data()
# In seq2seq, target is both input (shifted) and label. Here simplified.
model.fit([x_src, x_tgt], x_tgt, epochs=5, batch_size=32, verbose=1)
print("✅ Keras Training Complete.")
