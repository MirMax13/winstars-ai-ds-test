import tensorflow as tf
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from PIL import Image
import numpy as np

# NER
NER_MODEL_PATH = "./ner_model"
ner_tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_PATH)
ner_model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_PATH)

# Image classification
IMG_MODEL_PATH = "./cnn/animal_cnn_model.keras"
img_model = tf.keras.models.load_model(IMG_MODEL_PATH)
CLASS_NAMES = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spider', 'squirrel']

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
    img = Image.open(image_path).convert("RGB").resize((128, 128))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    preds = img_model.predict(img)
    return CLASS_NAMES[np.argmax(preds)]

def check_statement(text, image_path):
    text_animal = extract_animal_from_text(text)
    image_animal = classify_image(image_path)

    if text_animal is None:
        print("No animal found in text.")
        return None

    text_lower = text.lower()
    is_negated = any(word in text_lower for word in ["not", "no", "isn't", "aren't", "doesn't"])

    if is_negated:
        result = text_animal != image_animal
    else:
        result = text_animal == image_animal

    print(f"📝 Text animal: {text_animal}")
    print(f"🖼️ Image animal: {image_animal}")
    print(f"🧩 Negation: {is_negated}")
    print(f"✅ Result: {result}")
    return result

if __name__ == "__main__":
    text_input = "It's not a cow."
    image_path = "./dataset/test/cat/13.jpeg"
    check_statement(text_input, image_path)
