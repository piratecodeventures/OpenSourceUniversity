import torch
import torch.nn as nn
from torchvision import models
from transformers import DistilBertModel, DistilBertConfig

class DualEncoder(nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()
        
        # 1. Image Encoder (ResNet50)
        # We remove the final classification layer
        resnet = models.resnet50(pretrained=True)
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1]) # Output: 2048 dim
        self.image_projector = nn.Linear(2048, embedding_dim)
        
        # 2. Text Encoder (DistilBERT)
        # We use a small, fast Transformer
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.text_projector = nn.Linear(768, embedding_dim)
        
        # 3. Temperature (for Contrastive Loss scaling)
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592) # log(1/0.07)

    def encode_image(self, images):
        # images: [Batch, 3, 224, 224]
        features = self.image_encoder(images) # [Batch, 2048, 1, 1]
        features = features.flatten(1)         # [Batch, 2048]
        embeddings = self.image_projector(features)
        
        # Normalize embeddings (Crucial for Cosine Similarity)
        return features / features.norm(dim=-1, keepdim=True)

    def encode_text(self, input_ids, attention_mask):
        # Pass through BERT
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Take the [CLS] token (first token) as the semantic representation
        cls_token = output.last_hidden_state[:, 0, :] # [Batch, 768]
        embeddings = self.text_projector(cls_token)
        
        return embeddings / embeddings.norm(dim=-1, keepdim=True)

    def forward(self, images, input_ids, attention_mask):
        image_embeds = self.encode_image(images)
        text_embeds = self.encode_text(input_ids, attention_mask)
        return image_embeds, text_embeds

if __name__ == "__main__":
    print("Initializing Dual Encoder...")
    model = DualEncoder()
    print("✅ Model Built successfully.")
