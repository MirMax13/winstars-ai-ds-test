from tensorflow.keras.preprocessing.image import ImageDataGenerator
from classifier import AnimalCNNClassifier
import argparse

def main():
    parser = argparse.ArgumentParser(description="Train CNN model for animal image classification.")
    parser.add_argument("--train_dir", type=str, required=True, help="Path to training data directory.")
    parser.add_argument("--test_dir", type=str, required=True, help="Path to test data directory.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--img_size", type=int, default=128, help="Image size for resizing.")
    parser.add_argument("--output_path", type=str, default="animal_cnn_model.keras", help="Path to save trained model.")
    parser.add_argument("--load_model", type=str, help="Path to load existing model (keras).")
    args = parser.parse_args()

    IMG_SIZE = args.img_size
    BATCH_SIZE = args.batch_size

    # --- Generators ---
    train_gen = ImageDataGenerator(rescale=1./255, rotation_range=20, horizontal_flip=True)
    test_gen = ImageDataGenerator(rescale=1./255)

    train_ds = train_gen.flow_from_directory(
    args.train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
    )

    val_ds = test_gen.flow_from_directory(
        args.test_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE
    )
    # --- Model ---
    if args.load_model:
        model = AnimalCNNClassifier.load(args.load_model)
        print(f"Loaded model from {args.load_model}")
    else:
        model = AnimalCNNClassifier(img_size=IMG_SIZE, num_classes=train_ds.num_classes)

    history = model.train(train_ds, val_ds, epochs=args.epochs)

    model.save(args.output_path)
    print(f"Model saved to {args.output_path}")

if __name__ == "__main__":
    main()