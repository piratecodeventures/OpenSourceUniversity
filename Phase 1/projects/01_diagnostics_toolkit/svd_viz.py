import numpy as np
import matplotlib.pyplot as plt

def generate_image(size=256):
    # Create a synthetic grid/circle image
    x = np.linspace(-10, 10, size)
    y = np.linspace(-10, 10, size)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X) + np.cos(Y)
    return Z

def compress_svd(img, k):
    # 1. Decompose
    U, s, Vt = np.linalg.svd(img, full_matrices=False)
    
    # 2. Truncate (Keep top k singular values)
    U_k = U[:, :k]
    s_k = np.diag(s[:k])
    Vt_k = Vt[:k, :]
    
    # 3. Reconstruct
    return np.dot(U_k, np.dot(s_k, Vt_k)), s

def run_demo():
    img = generate_image()
    
    ks = [5, 20, 50, 256]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    
    for ax, k in zip(axes, ks):
        reconstruction, sigma = compress_svd(img, k)
        ax.imshow(reconstruction, cmap='gray')
        
        # Calculate compression ratio (naive)
        original_size = img.size
        compressed_size = k*img.shape[0] + k + k*img.shape[1]
        ratio = original_size / compressed_size
        
        ax.set_title(f"k={k}\nRatio: {ratio:.1f}x")
        ax.axis('off')
        
    plt.suptitle("SVD Image Compression At Different Ranks")
    plt.show()
    
    # Plot Singular Values (The Scree Plot)
    _, s, _ = np.linalg.svd(img)
    plt.figure()
    plt.plot(np.log(s))
    plt.title("Log Singular Values (Energy)")
    plt.xlabel("Index")
    plt.ylabel("Log Magnitude")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    run_demo()
