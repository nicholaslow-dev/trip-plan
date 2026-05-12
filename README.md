# SPECIALIZING SMALL LANGUAGE MODEL FOR RELIABLE TRIP PLANNING IN TOURISM

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue)](https://huggingface.co/NickolasLow1)

This repository contains the end-to-end research and implementation for specializing Small Language Models (SLMs) to solve complex, multi-city trip planning tasks. The project transitions from identifying failure modes in generalist models (Gemini) to developing a specialized Qwen 2.5 7B Coder agent that translates natural language into deterministic Operations Research (OR-Tools) code.

## 🚀 Key Features

- **Code-Generation Architecture**: Bypasses LLM mathematical reasoning flaws by using the model as a translator for Google OR-Tools.
- **Advanced RLVR Training**: Utilizes GRPO (Group Relative Policy Optimization) with verifiable execution rewards.
- **Deterministic Verification**: Integrated Python compiler to validate itineraries against hard constraints in real-time.
- **Stochastic Optimization**: Implements Pass@K evaluation to achieve near-frontier accuracy with a 7B parameter model.

## 📂 Project Structure

The project follows a research-centric organization where each directory corresponds to a specific **Research Objective (RO)**. 

### Core Directory Pattern
Every module in this repository is divided into two primary sub-folders to ensure a clean separation of concerns:
- **`data/`**: Dedicated storage for input datasets, intermediate results, and evaluation JSONs.
- **`src/`**: Contains all source code (Python scripts, Jupyter notebooks), visualizations, and logic.

### Research Phases

#### Phase 1: Baseline Setup & Evaluation
- **Location**: `p1_baseline-setup-and-evaluation`
- **RO1: Baseline Execution**: Establishing performance baselines using Gemini 1.5 Pro and base Qwen 2.5 models.
- **RO2: Error Analysis**: Qualitative error categorization (Logistical Hallucination, Contextual Amnesia).

#### Phase 2: Specialized SLM Development
- **Location**: `p2_specialized-slm-development-pipeline`
- **RO3: Supervised Fine-Tuning (SFT)**: Aligning Qwen 2.5 7B Coder with OR-Tools Python syntax using QLoRA.
- **RO4: Reinforcement Learning (GRPO)**: Optimizing functional correctness through execution-based rewards.

#### Phase 3: Final Evaluation & Trade-off Analysis
- **Location**: `p3_final-evaluation-and-trade-off-analysis`
- **RO5: Comparative Testing**: Specialized Agent vs. Gemini 1.5 Flash on held-out test data.
- **RO6: Trade-off Modeling**: Analysis of "Hidden API Costs," latency, and token verbosity.

## 📊 Performance Summary

| Model                    | Method                  | Accuracy   | Latency (Avg) |
| :----------------------- | :---------------------- | :--------- | :------------ |
| **Gemini 1.5 Flash**     | Zero-Shot NL            | 97.50%     | 51.35s        |
| **Specialized SLM (7B)** | Pass@1 (Greedy)         | 79.38%     | **49.02s**    |
| **Specialized SLM (7B)** | **Pass@5 (Stochastic)** | **86.88%** | 71.46s        |

## 🛠️ Environment & Setup

- **Hardware**: Tested on NVIDIA RTX 6000 Ada (48GB VRAM).
- **Environment**: Designed for the **Unsloth latest Docker** image.
- **Dependencies**: `unsloth`, `trl`, `transformers`, `ortools`, `vllm`.

```bash
# Clone the repository
git clone https://github.com/nicholaslow-dev/trip-plan.git
cd trip-plan

# Install core dependencies
pip install unsloth ortools trl transformers vllm
```

## 🤗 Model Checkpoints

The specialized models are hosted on Hugging Face for direct use:

- **SFT Checkpoint**: [NickolasLow1/sft-checkpoints](https://huggingface.co/NickolasLow1/sft-checkpoints/tree/main/checkpoint-126) (Syntax Saturation)
- **GRPO Checkpoint**: [NickolasLow1/grpo_train](https://huggingface.co/NickolasLow1/grpo_train/tree/main/checkpoint-240) (Optimal Reasoning Policy)

## 📜 License

Licensed under the [MIT License](LICENSE).

## 🤝 Acknowledgments

Developed as a Master's Capstone project for the **Asia Pacific University of Technology & Innovation (APU)**, Kuala Lumpur, Malaysia.
