import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 1. The Loss Function (The Landscape)
def loss_function(x, y):
    # A simple bowl with a saddle point
    return x**2 + 2*y**2 - 0.5*x*y

def gradients(x, y):
    # Analytical derivatives
    df_dx = 2*x - 0.5*y
    df_dy = 4*y - 0.5*x
    return df_dx, df_dy

# 2. Gradient Descent Optimizer
class Optimizer:
    def __init__(self, start_x, start_y, lr=0.1, momentum=0.9):
        self.x = start_x
        self.y = start_y
        self.lr = lr
        self.momentum = momentum
        self.vx = 0.0
        self.vy = 0.0
        self.history = []

    def step(self):
        # Calculate Gradient
        grad_x, grad_y = gradients(self.x, self.y)
        
        # Apply Momentum
        self.vx = self.momentum * self.vx - self.lr * grad_x
        self.vy = self.momentum * self.vy - self.lr * grad_y
        
        # Update Position
        self.x += self.vx
        self.y += self.vy
        
        self.history.append((self.x, self.y))
        return self.x, self.y

# 3. Visualization
def run_viz():
    print("Initializing Gradient Descent Simulation...")
    opt = Optimizer(start_x=4.0, start_y=3.0, lr=0.05, momentum=0.9)
    
    # Pre-calculate steps
    for _ in range(50):
        opt.step()
        
    path = np.array(opt.history)
    
    # Setup Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    x_range = np.linspace(-5, 5, 100)
    y_range = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = loss_function(X, Y)
    
    # Contour Map
    ax.contour(X, Y, Z, levels=20, cmap='viridis')
    line, = ax.plot([], [], 'ro-', label='SGD Path')
    
    def update(frame):
        line.set_data(path[:frame, 0], path[:frame, 1])
        return line,

    ani = FuncAnimation(fig, update, frames=len(path), interval=100)
    plt.title(f"Gradient Descent with Momentum (lr={opt.lr})")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_viz()
