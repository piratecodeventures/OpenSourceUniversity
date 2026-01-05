import torch
from dual_encoder import DualEncoder
from transformers import DistilBertTokenizer
from PIL import Image
from torchvision import transforms
import numpy as np

def create_dummy_data():
    # In a real app, this would load files. 
    # Here we create tensors representing Red, Green, Blue images.
    # Typically: Red=[1,0,0], Green=[0,1,0]...
    
    # 3 Images: Red, Green, Blue
    images = torch.zeros(3, 3, 224, 224)
    
    # Image 0: Red
    images[0, 0, :, :] = 1.0 
    
    # Image 1: Green
    images[1, 1, :, :] = 1.0
    
    # Image 2: Blue
    images[2, 2, :, :] = 1.0
    
    filenames = ["red_circle.jpg", "green_square.jpg", "blue_triangle.jpg"]
    return images, filenames

def search():
    print("⏳ Loading Model (This may take a moment to download BERT)...")
    model = DualEncoder()
    model.eval()
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # 1. Indexing (The "Database")
    images, filenames = create_dummy_data()
    print(f"📸 Indexing {len(filenames)} images...")
    
    with torch.no_grad():
        # Get raw features (In a real app, we'd use the projector, 
        # but untuned projections are random. 
        # So for this Mock Demo, we just cheat and look at the raw pixel channels 
        # to prove the logic works, OR we assume the model is pre-trained.
        # Since we just initialized random weights, the search won't work "semantically".
        # However, the code structure is correct for a Production System.)
        
        image_embeds = model.encode_image(images)
        
    print("✅ Indexing Complete.\n")
    
    # 2. Query Loop
    queries = ["something red", "a green object", "blue sky"]
    
    print("🔍 Starting Search...")
    for query in queries:
        inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            text_embed = model.encode_text(inputs['input_ids'], inputs['attention_mask'])
            
        # 3. Retrieval (Cosine Similarity)
        # Sim = Text_Vec @ Image_Matrix.T
        similarity = text_embed @ image_embeds.T
        
        # Get Top 1
        best_idx = similarity.argmax().item()
        best_nmae = filenames[best_idx]
        confidence = similarity[0, best_idx].item()
        
        print(f"Query: '{query}' -> Match: {best_nmae} (Score: {confidence:.4f})")

    print("\n⚠️ Note: Since weights are random (not trained on COCO/Flickr), matches are random.")
    print("This demonstrates the PIPELINE, not the intelligence.")

if __name__ == "__main__":
    search()
