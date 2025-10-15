from mnist_classifier_interface import MnistClassifierInterface
import tensorflow as tf

class CnnClassifier(MnistClassifierInterface):
    def __init__(self, model):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def train(self, train_data, train_labels):
        self.model.fit(train_data, train_labels, epochs=5)

    def predict(self, test_data):
        return self.model.predict(test_data)