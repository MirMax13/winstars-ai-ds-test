from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import argparse
import json

ANIMALS = ["dog", "horse", "elephant", "butterfly", "chicken", "cat", "cow", "sheep", "squirrel", "spider"]


def preprocess_text(text):
    return text.replace(".", "").replace(",", "").replace("'", "").replace("!", "").replace("?", "")

def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    return tokenizer, model

def predict_entities(text, tokenizer, model):
    text_cleaned = preprocess_text(text)
    inputs = tokenizer(text_cleaned, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)

    predictions = torch.argmax(outputs.logits, dim=2)
    predicted_labels = [model.config.id2label[pred.item()] for pred in predictions[0]]
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    detected_animals = []
    for token, label in zip(tokens, predicted_labels):
        if label == "B-ANIMAL" and token not in ["[CLS]", "[SEP]", "[PAD]"]:
            clean_token = token.replace("##", "").lower()
            detected_animals.append(clean_token)

    return detected_animals

def check_animal_in_text(text):
    text_lower = text.lower()
    for animal in ANIMALS:
        if animal in text_lower:
            return animal
    return None

def main():
    parser = argparse.ArgumentParser(description="NER model inference script")
    parser.add_argument('--model_path', type=str, default='./ner/model', help='Path to the trained NER model')
    parser.add_argument("--text", type=str, help="Single input text for prediction")
    parser.add_argument("--file", type=str, help="Path to file with multiple lines of text")
    parser.add_argument("--output", type=str, help="Path to save output JSON (optional)")
    args = parser.parse_args()

    tokenizer, model = load_model(args.model_path)
    results = {}

    if args.text:
        detected = predict_entities(args.text, tokenizer, model)
        results[args.text] = detected

    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            sentences = [line.strip() for line in f if line.strip()]
        for sentence in sentences:
            detected = predict_entities(sentence, tokenizer, model)
            results[sentence] = detected
    else:
        raise ValueError("Please provide either --text or --file argument.")

    # print or save
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"✅ Results saved to {args.output}")
    else:
        for sentence, detected in results.items():
            print(f"Text: '{sentence}'")
            print(f"  Detected: {detected if detected else 'None'}\n")

if __name__ == "__main__":
    main()