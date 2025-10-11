from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from datasets import load_dataset

labels = ["O", "B-ANIMAL", "I-ANIMAL"]
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}

model_name = "distilbert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(labels), id2label=id2label, label2id=label2id)

dataset = load_dataset("json", data_files="ner/data/ner_dataset.json")["train"]

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
    output_dir="./ner_model",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets,
    data_collator=data_collator,
)

trainer.train()

model.save_pretrained("./ner_model")
tokenizer.save_pretrained("./ner_model")

print("✅ NER model trained and saved.")