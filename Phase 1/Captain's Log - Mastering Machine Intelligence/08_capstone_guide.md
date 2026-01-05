# The Final Capstone: Sprint 14 (The Mastery)

## 📜 Story Mode: The Captain's Test

> **Mission Date**: 2043.12.31
> **Officers**: All Specialists
>
> **The Challenge**: You have learned the theory. You have built the components.
> Now, you must build the *Ship*.
>
> This is not a drill.
> You must choose your Path. You must define your own scope. You must deliver a working system.
> The universe doesn't grade on a curve. It works, or it doesn't.
>
> **Objective**: Build a production-grade AI System.
> **Constraint**: 4 Weeks. Zero Hand-holding.

---

## 🛤️ Choose Your Path

Select **ONE** of the following tracks for your final deliverable.

### Path A: The Enterprise LLM System (GenAI Specialist)
**Goal**: Build a vertical-specific LLM Application (e.g., Legal Analyst, Medical Assistant, Code Reviewer).

**Requirements**:
1.  **Fine-Tuning**: Must implement LoRA/PEFT on a 7B+ model (Llama 3, Mistral) on custom data.
2.  **RAG**: Must integrate a Vector DB (Pinecone/Weaviate) for retrieving facts.
3.  **Evaluation**: Must include an automated eval pipeline (RAGAS or LLM-as-Judge) measuring Accuracy and Hallucination rate.
4.  **UI**: A clean Streamlit/Chainlit interface.

**Hard Mode Challenge**:
*   Implement "Agentic" behavior (the model uses tools to verify its own answers).

---

### Path B: The Autonomous Agent (Robotics Specialist)
**Goal**: Build a system that navigates or manipulates its environment.

**Requirements**:
1.  **Simulation**: A robust PyBullet/Isaac Gym environment.
2.  **Perception**: A Vision module (CNN/ViT) that processes raw pixels.
3.  **Control**: An RL Policy (PPO/SAC) or Planner (RRT*) that achieves a goal.
4.  **Sim-to-Real**: Strategy for domain randomization or robust control.

**Hard Mode Challenge**:
*   Deploy the policy to a real Edge Device (Raspberry Pi/Jetson) or demonstrate "Visual Servoing".

---

### Path C: The ML Platform (Systems Specialist)
**Goal**: Build the infrastructure that *runs* the other two.

**Requirements**:
1.  **Distributed Training**: A pipeline that scales to multiple "devices" (can be simulated with Docker containers).
2.  **Feature Store**: An online/offline store guaranteeing consistency.
3.  **Serving Mesh**: A high-throughput API gateway (FastAPI/Triton) with Canary Deployment logic.
4.  **Drift Monitoring**: Automated alerts for data shift.

**Hard Mode Challenge**:
*   Implement a "Serverless" inference scaler that scales from 0 to N replicas based on request load (Horizontal Pod Autoscaler).

---

## 🏆 The Rubric (Success KPIs)

Your project will be graded on **The 4 Ps**:

1.  **Performance** (30%):
    *   Does it work? (High Accuracy / High Reward / Low Latency).
    *   Are the metrics rigorous?
2.  **Production** (30%):
    *   Is it Dockerized?
    *   Is there CI/CD?
    *   Is the code clean and typed?
3.  **Presentation** (20%):
    *   Is the Readme clear?
    *   Is there a Demo Video?
4.  **Passion** (20%):
    *   Did you go beyond the tutorial? (Complexity rating).

---

## 📦 Deliverables Checklist

- [ ] **GitHub Repo**: Clean, documented usage.
- [ ] **Technical Report**: A blog post explaining your Architecture decisions (Why LoRA? Why PPO?).
- [ ] **Demo**: A 2-minute video walkthrough.

---

> **Final Words**:
> "The code you write today will run the world tomorrow. Make it robust. Make it fair. Make it Matter."

*Good luck, Specialist.*
