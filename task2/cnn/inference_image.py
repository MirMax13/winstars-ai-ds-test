import tensorflow as tf
import numpy as np
import argparse
import json
import os

CLASS_NAMES = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider", "squirrel"]

def load_and_preprocess_image(image_path, img_size):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_size, img_size))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

def predict_image(model_path, image_path, img_size):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model not found at {model_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ Image not found at {image_path}")

    model = tf.keras.models.load_model(model_path)
    print(f"✅ Model loaded from {model_path}")

    img_batch = load_and_preprocess_image(image_path, img_size)

    preds = model.predict(img_batch)
    pred_idx = np.argmax(preds, axis=1)[0]
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(np.max(preds))

    return {"predicted_class": pred_class, "confidence": confidence}

def main():
    parser = argparse.ArgumentParser(description="Inference script for animal image classification model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model (.keras file)")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--img_size", type=int, default=128, help="Input image size (default=128)")
    parser.add_argument("--output", type=str, help="Optional: path to save JSON results")

    args = parser.parse_args()
    result = predict_image(args.model_path, args.image_path, args.img_size)

    print(f"\n🔍 Prediction result:")
    print(json.dumps(result, indent=2))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"✅ Results saved to {args.output}")

if __name__ == "__main__":
    main()
