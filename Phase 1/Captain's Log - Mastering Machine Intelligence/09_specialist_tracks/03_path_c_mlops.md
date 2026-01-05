# Path C: MLOps & Systems Specialist (The Commander)

## 📜 Career Profile: The Commander
*   **Role**: MLOps Engineer, AI Platform Architect, Reliability Engineer.
*   **Mission**: Build the "Rails" that AI runs on. Scale from 1 req/sec to 1M req/sec.
*   **Income Potential**: $170k - $450k (High stability).
*   **Core Stack**: Kubernetes, Ray, Terraform, Prometheus, Redis, Kafka.

---

## 📅 Sprint 11: Scalable Infrastructure (Weeks 41-44)

> **Theme**: "The Cluster"

### 11.1 Distributed Training Architectures
*   **11.1.1 The Bottlenecks**: Network Bandwidth (All-Reduce), GPU VRAM, IOPS.
*   **11.1.2 Parallelism Types**:
    *   **Data Parallel (DDP)**: Syncing gradients. Ring-AllReduce algo.
    *   **Pipeline Parallel (GPipe)**: Splitting layers across GPUs ($L_1 \to GPU_1, L_2 \to GPU_2$). Bubbles in pipeline.
    *   **Tensor Parallel (Megatron)**: Splitting matrix multiplication across GPUs.
*   **11.1.3 Frameworks**: Ray Train, PyTorch Lightning, DeepSpeed.

### 11.2 Containerization & Orchestration
*   **11.2.1 Docker Advanced**:
    *   Multi-stage builds (Build in Golang image, Copy binary to Alpine).
    *   Distroless images (Security).
    *   GPU Passthrough (NVIDIA Container Toolkit).
*   **11.2.2 Kubernetes (K8s)**:
    *   **Nodes & Pods**: The atomic unit.
    *   **Services**: ClusterIP, NodePort, LoadBalancer.
    *   **Ingress**: NGINX Controller.
    *   **CRDs (Custom Resource Definitions)**: TFJob, PyTorchJob (Kubeflow).

### 11.3 Infrastructure as Code (IaC)
*   **11.3.1 Terraform**: HCL (HashiCorp Language). Defining AWS VPCs, EC2s, and S3 buckets declaratively.
*   **11.3.2 Ansible**: Configuration management. Setting up CUDA drivers on 100 machines.

---

## 📅 Sprint 12: ML Platform Engineering (Weeks 45-48)

> **Theme**: "The Factory"

### 12.1 The Feature Store
*   **12.1.1 Why?**: The "Training-Serving Skew". Python features in training != Java features in prod.
*   **12.1.2 Architecture**:
    *   **Offline Store**: Cold storage (S3/BigQuery/Parquet) for batch training.
    *   **Online Store**: Hot storage (Redis/DynamoDB) for low-latency inference.
    *   **Sync**: Materialization jobs (Airflow) moving data Offline $\to$ Online.

### 12.2 Model Registry & CI/CD
*   **12.2.1 Versioning**: Tracking Code (Git), Data (DVC), and Model Artifact (MLflow).
*   **12.2.2 Promotion Gates**:
    *   Dev $\to$ Staging (Automated Tests).
    *   Staging $\to$ Prod (Manual Approval / Canary).
*   **12.2.3 Shadow Deployment**:
    *   Running new model V2 alongside V1 in prod.
    *   V2 receives traffic but V1 returns the response.
    *   Log V2 predictions to check without risk.

### 12.3 Workflow Orchestration
*   **12.3.1 Airflow**: DAGs. Scheduling Retraining jobs. Sensors.
*   **12.3.2 Kubeflow Pipelines**: ML-native workflows on K8s.

---

## 📅 Sprint 13: Edge AI & Optimization (Weeks 49-52)

> **Theme**: "Efficiency"

### 13.1 Model Compression
*   **13.1.1 Quantization**:
    *   **PTQ (Post-Training)**: Calibration on small dataset.
    *   **QAT (Quantization Aware Training)**: Simulating quantization noise during training.
*   **13.1.2 Pruning**: Structured (Channels) vs Unstructured (Weights). Sparse acceleration.
*   **13.1.3 Knowledge Distillation**: Teacher (Large) $\to$ Student (Small) using KL-Divergence loss.

### 13.2 Hardware Accelerators
*   **13.2.1 ONNX Runtime**: Graph optimizations (Constant folding, Operator fusion).
*   **13.2.2 TensorRT (NVIDIA)**: Kernel auto-tuning for specific GPU.
*   **13.2.3 Edge TPU (Coral)**: INT8 dedicated silicon.

---

## 📅 Sprint 14: Capstone - The Enterprise Platform (Weeks 53-56)

### 14.1 Observability & Monitoring
*   **14.1.1 The 3 Pillars**:
    *   **Metrics** (Prometheus): "CPU is 90%".
    *   **Logs** (ELK Stack): "Error: NullPointer on line 40".
    *   **Traces** (Jaeger): "DB Query took 500ms".
*   **14.1.2 ML Specific**:
    *   **Data Drift**: $P(X)$ changes. (e.g., Users get younger).
    *   **Concept Drift**: $P(Y|X)$ changes. (e.g., "Corona" beer searches span Viral).

### 14.2 Auto-Scaling Strategies
*   **14.2.1 HPA (Horizontal Pod Autoscaler)**: CPU/Memory based scaling.
*   **14.2.2 KEDA (Event-Driven)**: Scale based on "Kafka Queue Length" or "Pending Requests".

---

## 💻 The Ship's Code: Ray Auto-Scaling Deployment

```python
from ray import serve
from fastapi import FastAPI
import torch

app = FastAPI()

@serve.deployment(
    num_replicas="auto",
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 20,
        "target_num_ongoing_requests_per_replica": 5, # Scale trigger
        "upscale_delay_s": 2, # Responsive scaling
        "downscale_delay_s": 30 # Prevent flapping
    },
    ray_actor_options={"num_gpus": 0.5} # Packing 2 models per GPU
)
@serve.ingress(app)
class ModelService:
    def __init__(self):
        # Load optimized ONNX model
        self.session = onnxruntime.InferenceSession("model_quantized.onnx")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    @app.post("/predict")
    def predict(self, text: str):
        # 1. Preprocess
        inputs = self.tokenizer(text, return_tensors="np")
        
        # 2. Inference
        outputs = self.session.run(None, dict(inputs))
        
        # 3. Log for Monitoring (Async)
        self.log_prediction(text, outputs)
        
        return {"logit": float(outputs[0][0])}

serve.run(ModelService.bind())
```

---

## ❓ Industry Interview Corner

**Q1: "Explain the CAP Theorem and how it applies to Feature Stores."**
*   **Answer**: "Consistency, Availability, Partition Tolerance. Pick 2. **Online Stores** (Redis/Dynamo) prioritize AP (Always available for inference, eventual consistency). **Offline Stores** (HDFS) prioritize CP (Strong consistency for training data)."

**Q2: "How does Gradient Accumulation work, and when do you use it?"**
*   **Answer**: "When Batch Size > GPU VRAM. We run $N$ mini-batches, summing gradients *without* updating weights. Once $N$ are done, we call `optimizer.step()`. Simulates a large batch size on small hardware. Trade-off: Slower training."

**Q3: "What is Reservoir Sampling?"**
*   **Answer**: "A streaming algorithm to choose a random sample of $k$ items from a list of unknown length $n$. Essential for monitoring infinite streams of production data without storing everything."
