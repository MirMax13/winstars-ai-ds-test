"""
NER Model Training Script

This script fine-tunes a DistilBERT model for animal entity recognition.
The model learns to identify animal mentions in text with BIO tagging.

Labels:
- O: Outside (not an animal)
- B-ANIMAL: Beginning of animal entity
- I-ANIMAL: Inside animal entity (continuation)

Usage:
    python train_ner.py --data_path ./data/ner_dataset.json --output_dir ./model --epochs 3
"""

from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from datasets import load_dataset, Dataset
import json
import argparse

# Define NER labels using BIO tagging scheme
labels = ["O", "B-ANIMAL", "I-ANIMAL"]
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}

# Load base model and tokenizer
model_name = "distilbert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(
    model_name, 
    num_labels=len(labels), 
    id2label=id2label, 
    label2id=label2id
)

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train NER model for animal entity recognition")
parser.add_argument('--data_path', type=str, default='ner/data/ner_dataset.json', help='Path to the NER dataset JSON file')
parser.add_argument('--output_dir', type=str, default='./ner_model', help='Directory to save the trained model')
parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=8, help='Training batch size')
parser.add_argument('--logs_dir', type=str, default='./logs', help='Directory for training logs')
args = parser.parse_args()

# Load dataset
dataset = load_dataset("json", data_files=args.data_path)["train"]


def tokenize_and_align_labels(example):
    """
    Tokenize text and align labels with subword tokens.
    
    BERT tokenizers split words into subwords. This function ensures
    that labels are correctly aligned with the tokenized output.
    
    Args:
        example: Dataset example with 'tokens' and 'ner_tags' fields
        
    Returns:
        Tokenized inputs with aligned labels
    """
    tokenized_inputs = tokenizer(example["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    word_ids = tokenized_inputs.word_ids()
    
    for word_idx in word_ids:
        if word_idx is None:
            # Special tokens (CLS, SEP) get label -100 (ignored in loss)
            labels.append(-100)
        else:
            # Assign the label from the original word
            labels.append(label2id[example["ner_tags"][word_idx]])
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


# Tokenize the entire dataset
tokenized_datasets = dataset.map(tokenize_and_align_labels)

# Data collator for batching (handles padding)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# Configure training parameters
args = TrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=args.batch_size,
    num_train_epochs=args.epochs,
    logging_dir=args.logs_dir,
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets,
    data_collator=data_collator,
)

# Train the model
trainer.train()

# Save the trained model and tokenizer
model.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)

print("✅ NER model trained and saved.")