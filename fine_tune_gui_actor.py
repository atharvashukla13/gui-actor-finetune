from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
from gui_actor.model import GUIActorModel, ActionHead, GroundingVerifier
from gui_actor.dataset import GUIDataset
import yaml, os
import json


# Load config
with open("data_config.yaml") as f:
    cfg = yaml.safe_load(f)

# Load backbone and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutlmv3-base", use_fast=True)
print(f"Loaded tokenizer type: {type(tokenizer)}")

print("Loaded tokenizer type:", type(tokenizer))

backbone = AutoModel.from_pretrained(cfg["backbone_model"])

# Init custom model
model = GUIActorModel(
    backbone,
    ActionHead(hidden_size=backbone.config.hidden_size),
    GroundingVerifier(input_dim=backbone.config.hidden_size)
)

# Freeze backbone
for param in model.backbone.parameters():
    param.requires_grad = False

# Datasets
with open(cfg["train_data"], "r") as f:
    train_data = json.load(f)

with open(cfg["val_data"], "r") as f:
    val_data = json.load(f)

train_dataset = GUIDataset(train_data, tokenizer)
val_dataset = GUIDataset(val_data, tokenizer)

# Training arguments
training_args = TrainingArguments(
    output_dir="./checkpoints",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    learning_rate=5e-5,
    logging_dir="./logs",
    eval_strategy="epoch",      
    save_strategy="epoch",
    save_total_limit=2
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    # compute_metrics=your_metric_fn  # Optional
)

# Train (warm-up)
trainer.train()

# Unfreeze and continue
for param in model.backbone.parameters():
    param.requires_grad = True

# Resume fine-tuning
if os.path.exists("./checkpoints/checkpoint-last"):
    trainer.train(resume_from_checkpoint="./checkpoints/checkpoint-last")
else:
    trainer.train()
