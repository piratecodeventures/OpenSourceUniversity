import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms

def get_model(num_classes=2):
    print("⬇️ Downloading Pre-trained ResNet18...")
    # 1. Load Pre-trained Model
    model = models.resnet18(pretrained=True)
    
    # 2. Freeze all params (Transfer Learning)
    for param in model.parameters():
        param.requires_grad = False
        
    # 3. Replace the final Fully Connected layer
    # ResNet18's fc input size is 512
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

def train_dummy():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using Device: {device}")
    
    model = get_model().to(device)
    
    # Only optimize the Final Layer (fc)
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Dummy Data (Batch 8, 3 Channels, 224x224)
    inputs = torch.randn(8, 3, 224, 224).to(device)
    labels = torch.randint(0, 2, (8,)).to(device)
    
    print("🔄 Starting Training Loop (Simulated)...")
    model.train()
    for epoch in range(5):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/5 | Loss: {loss.item():.4f}")
        
    print("✅ Training Complete. Saving 'resnet_finetuned.pth'")
    torch.save(model.state_dict(), "resnet_finetuned.pth")

if __name__ == "__main__":
    train_dummy()
