import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from transformers import AutoTokenizer, AutoModelForTokenClassification
from PIL import Image
import os
import argparse
import sys

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Paths - use absolute paths relative to project structure
NER_MODEL_PATH = os.path.join(PROJECT_ROOT, "ner", "model")
IMG_MODEL_PATH = os.path.join(PROJECT_ROOT, "cnn", "resnet_model.pth")
IMG_STATS_PATH = os.path.join(PROJECT_ROOT, "cnn", "resnet_model_stats.pth")

# Italian to English mapping
ITALIAN_TO_ENGLISH = {
    "cane": "dog",
    "cavallo": "horse",
    "elefante": "elephant",
    "farfalla": "butterfly",
    "gallina": "chicken",
    "gatto": "cat",
    "mucca": "cow",
    "pecora": "sheep",
    "ragno": "spider",
    "scoiattolo": "squirrel",
}

# Load NER model
print("Loading NER model...")
ner_tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_PATH)
ner_model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_PATH)

# Load image classification model
print("Loading image classification model...")
device = "cuda" if torch.cuda.is_available() else "cpu"


class ConvertToRGB:
    def __call__(self, img):
        if img.mode != "RGB":
            return img.convert("RGB")
        return img


def create_resnet_model(num_classes=10):
    model = torchvision.models.resnet50(weights=None)
    in_features = model.fc.in_features
    classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )
    model.fc = classifier
    return model


# Load image model
img_model = create_resnet_model(num_classes=10)
checkpoint = torch.load(IMG_MODEL_PATH, map_location=device)
img_model.load_state_dict(checkpoint['model_state_dict'])
img_model = img_model.to(device)
img_model.eval()

# Load dataset stats (or use defaults)
if os.path.exists(IMG_STATS_PATH):
    stats = torch.load(IMG_STATS_PATH, map_location='cpu')
    mean = stats['mean']
    std = stats['std']
    class_names = stats['class_names']
else:
    print("Warning: Stats file not found, using default normalization")
    mean = torch.tensor([0.5177, 0.5003, 0.4125])
    std = torch.tensor([0.2659, 0.2610, 0.2785])
    class_names = sorted(ITALIAN_TO_ENGLISH.keys())

# Create transform
img_transform = transforms.Compose([
    ConvertToRGB(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

print("Models loaded successfully!")
print(f"Device: {device}")


def extract_animal_from_text(text: str):
    inputs = ner_tokenizer(text, return_tensors="pt")
    outputs = ner_model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=2)
    tokens = ner_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    animal = None
    for token, label_id in zip(tokens, predictions[0].numpy()):
        label = ner_model.config.id2label[label_id]
        if label == "B-ANIMAL":
            animal = token.replace("##", "")
            break
    return animal


def classify_image(image_path):
    img = Image.open(image_path)
    img_tensor = img_transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = img_model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)
    
    pred_idx = torch.argmax(probabilities, dim=1).item()
    italian_class = class_names[pred_idx]
    english_class = ITALIAN_TO_ENGLISH.get(italian_class, italian_class)
    
    return english_class


def check_statement(text, image_path):
    text_animal = extract_animal_from_text(text)
    image_animal = classify_image(image_path)

    if text_animal is None:
        print("No animal found in text.")
        return None

    text_lower = text.lower()
    # Extended negation detection based on training templates
    negation_patterns = [
        "not", "no", "isn't", "aren't", "doesn't", "don't",
        "lacks", "lack", "without",
        "can't find", "cannot find",
        "never", "none"
    ]
    is_negated = any(pattern in text_lower for pattern in negation_patterns)

    if is_negated:
        result = text_animal != image_animal
    else:
        result = text_animal == image_animal

    print(f"Text animal: {text_animal}")
    print(f"Image animal: {image_animal}")
    print(f"Negation: {is_negated}")
    print(f"Result: {result}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Animal Detection Pipeline - Verify if text statement matches image content")
    parser.add_argument("--text", type=str, required=True, help="Text statement about the animal (e.g., 'There is a dog in the picture')")
    parser.add_argument("--image", type=str, required=True, help="Path to the image file")
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        sys.exit(1)
    
    # Run the pipeline
    result = check_statement(args.text, args.image)
    
    # Print final result
    if result is None:
        print("\nFinal Result: Unable to verify (no animal found in text)")
        sys.exit(2)
    else:
        print(f"\nFinal Result: {result}")
        sys.exit(0 if result else 1)