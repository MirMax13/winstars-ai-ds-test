import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from PIL import Image
import warnings

warnings.simplefilter("ignore")

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


def load_model(checkpoint_path, num_classes=10, device='cuda'):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
    
    model = create_resnet_model(num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    if 'val_accuracy' in checkpoint:
        print(f"Validation accuracy: {checkpoint['val_accuracy']*100:.2f}%")
    
    return model


def load_dataset_stats(checkpoint_path):
    stats_path = checkpoint_path.replace('.pth', '_stats.pth')
    
    if os.path.exists(stats_path):
        stats = torch.load(stats_path, map_location='cpu')
        mean = stats['mean']
        std = stats['std']
        class_names = stats['class_names']
        print(f"Dataset statistics loaded from {stats_path}")
    else:
        print(f"Warning: Stats file not found, using default normalization")
        mean = torch.tensor([0.5, 0.5, 0.5])
        std = torch.tensor([0.5, 0.5, 0.5])
        class_names = sorted(ITALIAN_TO_ENGLISH.keys())
    
    return mean, std, class_names


def get_transform(mean, std):
    transform = transforms.Compose([
        ConvertToRGB(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return transform


def load_and_preprocess_image(image_path, transform):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    img = Image.open(image_path)
    img_tensor = transform(img)
    img_batch = img_tensor.unsqueeze(0)
    
    return img_batch


def predict_image(model, image_tensor, class_names, device='cuda'):
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
    
    confidence, pred_idx = torch.max(probabilities, dim=1)
    pred_idx = pred_idx.item()
    confidence = confidence.item()
    
    italian_class = class_names[pred_idx]
    english_class = ITALIAN_TO_ENGLISH.get(italian_class, italian_class)
    
    all_probs = probabilities[0].cpu().numpy()
    class_probabilities = {
        ITALIAN_TO_ENGLISH.get(cls, cls): float(prob)
        for cls, prob in zip(class_names, all_probs)
    }
    
    class_probabilities = dict(sorted(class_probabilities.items(), 
                                      key=lambda x: x[1], 
                                      reverse=True))
    
    result = {
        "predicted_class": english_class,
        "confidence": confidence,
        "all_probabilities": class_probabilities
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Inference script for animal image classification")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model (.pth file)")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default=None, help="Path to save JSON results")
    
    args = parser.parse_args()
    
    # Fixed parameters
    num_classes = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Device: {device}")
    
    model = load_model(args.model_path, num_classes=num_classes, device=device)
    mean, std, class_names = load_dataset_stats(args.model_path)
    transform = get_transform(mean, std)
    
    print(f"Loading image from {args.image_path}...")
    image_tensor = load_and_preprocess_image(args.image_path, transform)
    
    print(f"Running inference...")
    result = predict_image(model, image_tensor, class_names, device)
    
    print(f"\nPrediction result:")
    print(json.dumps(result, indent=2))
    
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {args.output}")
    
    return result


if __name__ == "__main__":
    main()
