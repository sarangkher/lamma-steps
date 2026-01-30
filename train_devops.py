import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset

# 1. PREPARE YOUR DATA
# We are creating a small dataset of DevOps intents
data = {
    "text": [
        "How do I restart Nginx on Ubuntu?",
        "Configure HashiCorp Vault for production",
        "Debug a blank page in Three.js",
        "Git command to revert the last commit",
        "Set up a reverse proxy on Nginx",
        "How to check system logs in Linux"
    ],
    "label": [0, 1, 2, 1, 0, 0] # 0: WebServer, 1: Security/DevOps, 2: Frontend
}

dataset = Dataset.from_dict(data)

# 2. LOAD TOKENIZER AND MODEL
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_func(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(tokenize_func, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# 3. CONFIGURE FOR GTX 970 (4GB VRAM)
training_args = TrainingArguments(
    output_dir="./devops_model",
    per_device_train_batch_size=4,   # Very small for 4GB VRAM
    num_train_epochs=5,
    fp16=True,                       # Use half-precision (crucial for GTX 970)
    logging_steps=10,
    learning_rate=5e-5,
    save_strategy="no"               # Don't waste disk space for this test
)

# 4. INITIALIZE TRAINER
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# 5. EXECUTE TRAINING
print("Starting training on GPU...")
trainer.train()
print("Success! Model trained.")
