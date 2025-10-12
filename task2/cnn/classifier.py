import tensorflow as tf

class AnimalCNNClassifier:
    def __init__(self, img_size=128, num_classes=10):
        self.IMG_SIZE = img_size
        self.num_classes = num_classes

        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✅ {len(gpus)} GPU(s) detected and memory growth enabled.")
            except RuntimeError as e:
                print(e)
        else:
            print("⚠️ No GPU detected, using CPU.")

        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.IMG_SIZE, self.IMG_SIZE, 3)),
            
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            # Classifier
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])

        self.model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

    def train(self, train_ds, val_ds, epochs=10):
        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs
        )
        return history

    def predict(self, img_batch):
        preds = self.model.predict(img_batch)
        return preds.argmax(axis=1)

    def save(self, path="animal_cnn_model"):
        self.model.save(path)
        print(f"✅ Model saved to {path}")

    def load(self, path="animal_cnn_model"):
        self.model = tf.keras.models.load_model(path)
        print(f"✅ Model loaded from {path}")
