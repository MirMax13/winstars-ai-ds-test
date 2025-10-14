from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from datasets import load_dataset, Dataset
import json
import argparse

labels = ["O", "B-ANIMAL", "I-ANIMAL"]
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}

model_name = "distilbert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(labels), id2label=id2label, label2id=label2id)

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='ner/data/ner_dataset.json', help='Path to the NER dataset JSON file')
parser.add_argument('--output_dir', type=str, default='./ner_model', help='Directory to save the trained model')
parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=8, help='Training batch size')
parser.add_argument('--logs_dir', type=str, default='./logs', help='Directory for training logs')
args = parser.parse_args()

dataset = load_dataset("json", data_files=args.data_path)["train"]

def tokenize_and_align_labels(example):
    tokenized_inputs = tokenizer(example["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    word_ids = tokenized_inputs.word_ids()
    for word_idx in word_ids:
        if word_idx is None:
            labels.append(-100)
        else:
            labels.append(label2id[example["ner_tags"][word_idx]])
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_and_align_labels)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

args = TrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=args.batch_size,
    num_train_epochs=args.epochs,
    logging_dir=args.logs_dir,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets,
    data_collator=data_collator,
)

trainer.train()

model.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)

print("✅ NER model trained and saved.")