import torch
import torch.nn as nn
import torch.optim as optim
import math

# 1. Config
SRC_VOCAB = 100
TGT_VOCAB = 100
D_MODEL = 32
HEADS = 2
LAYERS = 2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 2. Dummy Data Generator (English -> Alien)
# Alien Rule: Reverse list
def get_batch():
    # Batch of 2 sentences, length 5
    src = torch.randint(1, SRC_VOCAB, (2, 5)).to(DEVICE)
    tgt = torch.flip(src, [1]).to(DEVICE) # Reverse
    return src, tgt

# 3. Model Definition (PyTorch Style)
class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.src_embed = nn.Embedding(SRC_VOCAB, D_MODEL)
        self.tgt_embed = nn.Embedding(TGT_VOCAB, D_MODEL)
        self.pos_encoder = nn.Parameter(torch.randn(1, 100, D_MODEL))
        
        # The Core Transformer
        self.transformer = nn.Transformer(
            d_model=D_MODEL, nhead=HEADS, 
            num_encoder_layers=LAYERS, num_decoder_layers=LAYERS,
            batch_first=True
        )
        
        self.fc_out = nn.Linear(D_MODEL, TGT_VOCAB)
    
    def forward(self, src, tgt):
        # Embed + Positional
        src = self.src_embed(src) + self.pos_encoder[:, :src.shape[1], :]
        tgt = self.tgt_embed(tgt) + self.pos_encoder[:, :tgt.shape[1], :]
        
        # Masking (Causal Mask for Decoder)
        # 0s and -infs
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.shape[1]).to(DEVICE)
        
        # Pass through Transformer
        out = self.transformer(src, tgt, tgt_mask=tgt_mask)
        return self.fc_out(out)

# 4. Training Loop
model = TransformerModel().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print("🔥 PyTorch Training Start...")
for epoch in range(10):
    src, tgt = get_batch()
    
    # Teacher Forcing: Input to decoder is Shifted Right
    # But here for simplicity we just feed full tgt
    tgt_input = tgt 
    tgt_output = tgt # In real NMT, offset by 1
    
    optimizer.zero_grad()
    output = model(src, tgt_input)
    
    # Reshape for Loss: (Batch*Seq, Vocab)
    loss = criterion(output.view(-1, TGT_VOCAB), tgt_output.view(-1))
    loss.backward()
    optimizer.step()
    
    if epoch % 2 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

print("✅ PyTorch Training Complete.")
