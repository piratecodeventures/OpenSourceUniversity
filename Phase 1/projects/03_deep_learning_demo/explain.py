import torch
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def __call__(self, x, class_idx=None):
        # 1. Forward
        self.model.zero_grad()
        output = self.model(x)
        
        if class_idx is None:
            class_idx = torch.argmax(output)
            
        # 2. Backward (Target Class Score)
        score = output[0, class_idx]
        score.backward()
        
        # 3. Generate Map
        # Global Average Pooling of Gradients (Importance Weights)
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Weight the Activations
        activation = self.activations[0] # (512, 7, 7)
        for i in range(512):
            activation[i, :, :] *= pooled_gradients[i]
            
        # Average across channels
        heatmap = torch.mean(activation, dim=0).cpu().detach().numpy()
        
        # ReLU (We only care about Positive influence)
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize
        heatmap /= np.max(heatmap)
        return heatmap

def run_explainability():
    # Load Model
    model = models.resnet18(pretrained=True)
    model.eval()
    
    # Target Layer: Usually the last conv layer (layer4)
    # ResNet structure: layer1, layer2, layer3, layer4
    grad_cam = GradCAM(model, model.layer4[-1])
    
    # Dummy Image
    print("📸 Processing Dummy Image...")
    img_tensor = torch.randn(1, 3, 224, 224)
    
    # Generate Heatmap
    heatmap = grad_cam(img_tensor)
    
    # Resize to Image Size
    heatmap = cv2.resize(heatmap, (224, 224))
    
    # Plot
    plt.imshow(heatmap, cmap='jet')
    plt.title("Grad-CAM Heatmap (Where the model is looking)")
    plt.colorbar()
    plt.savefig("heatmap.png")
    print("✅ Heatmap saved to 'heatmap.png'")

if __name__ == "__main__":
    run_explainability()
