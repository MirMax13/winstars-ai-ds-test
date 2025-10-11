from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

model_path = "./ner_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)

ANIMALS = ["dog", "horse", "elephant", "butterfly", "chicken", "cat", "cow", "sheep", "squirrel", "spider"]


def preprocess_text(text):
    return text.replace(".", "").replace(",", "").replace("'", "").replace("!", "").replace("?", "")
def predict_entities(text):
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

test_sentences = [
    "I see a dog.",
    "There is a cat in the picture.",
    "The elephant is running.",
    "This is a car.",
    "Look, a butterfly!",
    "It's not a cow.",
    "I don't see a horse here.",
    "A wild spider appears.",
    "This is clearly not a chicken.",
]

print("🔍 NER Inference Results:\n")
print("=" * 70)

for sentence in test_sentences:
    expected_animal = check_animal_in_text(sentence)
    
    detected_animals = predict_entities(sentence)
    
    print(f"Text: '{sentence}'")
    print(f"  Expected: {expected_animal}")
    print(f"  Detected: {detected_animals if detected_animals else 'None'}")
