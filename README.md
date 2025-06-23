# GUI-Actor Fine-Tuning

This repository contains code to fine-tune the [GUI-Actor](https://microsoft.github.io/GUI-Actor/) model proposed by Microsoft Research, which enables intelligent agents to interact with graphical user interfaces through screen understanding and action prediction.

> ⚠️ Note: This repository includes code for fine-tuning only. Actual training is not performed here due to computational constraints.


## 📄 Paper Summary

**Title:** GUI-ACT: A Benchmark for Multi-step GUI Interaction Tasks

**Authors:** Luyu Gao, Chen Henry Wu, Morteza Ziyadi, Yiming Yang, et al.

**Highlights:**
- Introduces a large-scale dataset for GUI-based multi-step interaction.
- Proposes the GUI-ACT framework combining visual grounding and action prediction.
- Uses [LayoutLMv3](https://huggingface.co/microsoft/layoutlmv3-base) as the visual-textual backbone.


## 🧠 Repository Overview

gui-actor-finetune/
├── gui_actor/
│ ├── dataset.py # Dataset preprocessing logic
│ ├── model.py # GUIActorModel definition
│ └── utils.py # Utility functions
├── fine_tune_gui_actor.py # Main script for training
├── run_config.json # Training configuration (optional)
├── .gitignore # Prevents tracking large or unneeded files
└── README.md # You're here!


## 🚀 Features

- ✅ Modular model architecture with:
  - `LayoutLMv3` as backbone
  - Custom `ActionHead` for action classification
  - Custom `GroundingVerifier` for patch grounding scores
- ✅ Handles structured data including bounding boxes (`bbox`)
- ✅ Integrated with Huggingface `Trainer` for easy training and evaluation


## 🛠️ Setup Instructions

1. **Install dependencies:**
   pip install -r requirements.txt
Directory structure:
Ensure your dataset is preprocessed similarly to the GUI-ACT format, with input_ids, attention_mask, bbox, and labels.

Run training:

python fine_tune_gui_actor.py
Note: Large model checkpoints are excluded due to GitHub's 100MB limit.

📦 Model Components
GUIActorModel
Combines a transformer backbone (LayoutLMv3Model) with:

ActionHead: Classifies UI tokens into actions (e.g., click, type)

GroundingVerifier: Scores visual grounding between patches

ActionHead
Linear(hidden_size, 2) for binary action classification

GroundingVerifier
Small MLP that outputs a relevance score for each patch

🧪 Sample Training Logs
bash
Copy
Edit
Epoch 1: eval_loss = 0.6831
Epoch 2: eval_loss = 0.6811
Epoch 3: eval_loss = 0.6746
Train loss: 0.7388
🧠 References
Official Project Page: GUI-Actor

Paper: arXiv:2401.00055

LayoutLMv3: Huggingface Model Card

🙋‍♂️ Author
Atharva Shukla
GitHub: @atharvashukla13
For internship, research, or collaboration, feel free to reach out!

🛡️ License
This project is intended for educational and research use only
