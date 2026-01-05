# MLOps: Docker & Containers (The Ship)

## 📜 Story Mode: The Shipping Container

> **Mission Date**: 2043.08.01
> **Location**: Deep Space Outpost "Vector Prime"
> **Officer**: Lead Engineer Kael
>
> **The Problem**: I built the Universal Translator on my local workstation. It works perfectly.
> I sent the code to the remote server on Mars. It crashed.
> "Error: `torch` version mismatch."
> "Error: `CUDA` driver not found."
>
> I spent 3 days debugging dependencies. I fixed it.
> Then I moved it to the orbital station. It crashed again. "Error: Python 3.8 required."
>
> I am done fixing environments.
> I need a way to wrap my code, my libraries, and my OS into a sealed box.
> A box that runs exactly the same way on Earth, Mars, or orbit.
>
> *"Computer! Initialize Docker. Build Image. Ship it."*

---

## 1. Problem Setup & Motivation

### The 6 Engineering Questions
1.  **WHAT**:
    *   **Docker**: A platform to run applications in isolated environments called **Containers**.
    *   **Image**: The blueprint (Read-Only).
    *   **Container**: The running instance (Read-Write).
2.  **WHY**: Solves "It works on my machine". dependency hell.
3.  **WHEN**: Always. Development, Testing, Production.
4.  **WHERE**: `Dockerfile`, `docker build`.
5.  **WHO**: Solomon Hykes (2013).
6.  **HOW**: Uses Linux Namespaces and Cgroups to isolate processes.

> [!NOTE]
> **🛑 Pause & Explain (In Simple Words)**
>
> **The House vs The Hotel.**
>
> - **Virtual Machine (VM)**: Like building a whole new house. It has its own foundation, plumbing, and roof (Full OS Kernel). Heavy and slow.
> - **Container**: Like a hotel room. It shares the foundation and plumbing (Host OS Kernel) with other rooms, but inside the room, it looks like a private house. Lightweight and fast.

---

## 2. Mathematical Problem Formulation

### Isolation overhead
*   **VM**: Overhead $\approx$ 20% (Duplicated OS).
*   **Docker**: Overhead $\approx$ 1% (System calls are direct).
*   **Benefit**: You can run 10x more containers than VMs on the same hardware.

---

## 3. Step-by-Step Derivation

### The Dockerfile
Steps to build an image:
1.  **Base Image**: Start with a pre-built OS (`FROM python:3.9`).
2.  **Workdir**: Set up a folder (`WORKDIR /app`).
3.  **Copy**: Copy files (`COPY . .`).
4.  **Install**: Download libs (`RUN pip install -r requirements.txt`).
5.  **Command**: What to do when starting (`CMD ["python", "app.py"]`).

---

## 4. The Trifecta: Implementation Levels

We will write a **Production-Grade Dockerfile** for our NLP Project.

### The Ship's Code (Polyglot: Pure Python + Libraries)

```python
import os
import shutil

# LEVEL 0: Pure Python (The Concept of Isolation)
# Simulating what Docker does (Chroot)
def mock_container_run(command):
    # 1. Create limited filesystem (The Image)
    fs_root = "./mock_container_fs"
    os.makedirs(fs_root, exist_ok=True)
    
    # 2. "Copy" isolation (Mock)
    print(f"[Kernel] Isolating process in {fs_root}")
    print(f"[Kernel] Limiting RAM to 512MB (cgroups mock)")
    
    # 3. Exec
    # In Linux, we would do: os.chroot(fs_root)
    # Here we just pretend
    print(f"[Container] Running: {command}")
    
# LEVEL 1: The Basic Dockerfile
# Good for quick scripts
"""
FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install torch transformers flask
CMD ["python", "app.py"]
"""

# LEVEL 2: The Optimized-Layer Dockerfile (Cache)
# Docker caches layers. If you change code, you don't want to re-download PyTorch.
"""
FROM python:3.9-slim
WORKDIR /app

# 1. Copy Requirements FIRST
COPY requirements.txt .

# 2. Install (This layer is cached unless requirements.txt changes)
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy Code SECOND (This layer changes often)
COPY . .

CMD ["python", "app.py"]
"""
```

> [!CAUTION]
> **🛑 Production Warning**
>
> **The 10GB Image**:
> Adding `COPY data/` to your image makes it huge.
> **Fix**: Never put data in the image. Mount it as a **Volume** (`-v /host/data:/container/data`).

### Level 3: The Multi-Stage Build (Production Size)
*Reduce image size by throwing away build tools.*

```dockerfile
# Stage 1: Builder
FROM python:3.9-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Stage 2: Runner
FROM python:3.9-slim
WORKDIR /app
# Copy only installed packages from builder
COPY --from=builder /root/.local /root/.local
COPY . .
ENV PATH=/root/.local/bin:$PATH

CMD ["python", "app.py"]
```

> [!TIP]
> **👁️ Visualizing the Benefit: Layer Caching**
> Run this script to simulate how Order matters for build speed.
>
> ```python
> import matplotlib.pyplot as plt
> import numpy as np
>
> def plot_docker_layers():
>     # Scenario 1: Bad Order (Copy Code -> Install Libs)
>     # Code changes every commit. Cache valid: 0%. Rebuilds Libs every time.
>     t_bad = [100, 100, 100, 100, 100] # Seconds per build
>     
>     # Scenario 2: Good Order (Install Libs -> Copy Code)
>     # Libs change rarely. Cache valid: 90%. Rebuilds simple.
>     t_good = [100, 2, 2, 2, 2] # Seconds per build
>     
>     builds = [1, 2, 3, 4, 5]
>     
>     plt.figure(figsize=(10, 6))
>     plt.plot(builds, t_bad, 'r-o', label='Bad Dockerfile (No Cache)')
>     plt.plot(builds, t_good, 'g-o', label='Good Dockerfile (Layer Caching)')
>     
>     plt.title("Impact of Docker Layer Caching on CI/CD Speed")
>     plt.xlabel("Build Number (Over Time)")
>     plt.ylabel("Build Time (Seconds)")
>     plt.ylim(0, 120)
>     plt.legend()
>     plt.grid(True, linestyle='--', alpha=0.5)
>     
>     plt.text(1.5, 90, "Re-downloading PyTorch\nevery time :(", color='red')
>     plt.text(2.5, 10, "Using Cache :)", color='green')
>     
>     plt.show()
>
> # Uncomment to run:
> # plot_docker_layers()
> ```

---

## 5. System-Level Integration

```mermaid
graph LR
    Dev[Developer Laptop] -- Push Code --> Git
    Git -- Push --> CI[CI/CD Pipeline]
    CI -- Build --> Docker[Docker Image]
    Docker -- Push --> Registry[Docker Hub/ECR]
    Registry -- Pull --> Server[Production Server]
    Server -- Run --> Container
```

**Where it lives**:
**Kubernetes (K8s)**: An orchestrator that manages thousands of Docker containers.
**AWS Lambda**: Can run Docker images as functions.

---

## 6. Evaluation & Failure Analysis

### Failure Mode: The 10GB Image
Adding `COPY data/` to your image makes it huge.
**Fix**: Never put data in the image. Mount it as a **Volume** (`-v /host/data:/container/data`).

---

## 7. Ethics, Safety & Risk Analysis

### Root Access
By default, Docker containers run as Root.
If a hacker escapes the container, they own the host.
**Fix**: `USER appuser`. Always run as non-root.

---

## 8. Advanced Theory & Research Depth

### Distroless Images
Google's stripped-down images. Contains *only* your app and its dependencies. No shell, no `ls`, no `apt`.
Security through minimalism.

---

## 9. Assessment & Mastery Checks

### 13. Assessment & Mastery Checks

**Q1: Layer Caching**
Why is `COPY . .` usually the last step?
*   *Answer*: Because source code changes most frequently. If put earlier, it invalidates the cache for all subsequent steps (like `pip install`), causing slow rebuilds.

**Q2: Docker vs VM**
Why is Docker faster?
*   *Answer*: Docker shares the Host Kernel. It doesn't need to boot a new OS. VM emulates hardware + full OS.

**Q3: Orchestration**
What problem does Kubernetes solve that Docker doesn't?
*   *Answer*: "My container died, please restart it" and "I need 50 copies of this container". Docker runs IT. Kubernetes MANAGES it.

### 14. Common Misconceptions (Debug Your Thinking)

> [!WARNING]
> **"Docker guarantees reproducibility."**
> *   **Correction**: Only if you pin versions (`python:3.9` -> `python:3.9.12`). If you use `latest`, it might break tomorrow.

> [!WARNING]
> **"Containers are secure."**
> *   **Correction**: They share the kernel. A kernel exploit (Dirty COW) can break out of a container. VMs are safer for hostile code.

---

## 10. Further Reading & Tooling

*   **Tool**: **Docker Compose** (Run multi-container apps: App + Database).
*   **Concept**: **Kubernetes** (Orchestration).

---

## 11. Concept Graph Integration

*   **Previous**: [NLP Project](../projects/05_nlp_transformer/README.md).
*   **Next**: [Model Serving](05_mlops/02_serving.md).

### Concept Map
```mermaid
graph TD
    App[Application] --> Dependencies
    App --> OS[OS Libraries]
    
    Problem[Works on my Machine] -- "Solved by" --> Container
    
    VM[Virtual Machine] -- "Heavy" --> Hypervisor
    VM -- "Slow" --> GuestOS
    
    Container[Docker Container] -- "Light" --> Engine
    Container -- "Fast" --> SharedKernel
    
    Workflow --> Dockerfile[Blueprint]
    Dockerfile --> Build
    Build --> Image[Image (Read-Only)]
    Image --> Run
    Run --> Instance[Container (Read-Write)]
    
    Optimization --> Caching[Layer Caching]
    Optimization --> MultiStage[Multi-Stage Build]
    
    style VM fill:#faa,stroke:#333
    style Container fill:#afa,stroke:#333
```
```
