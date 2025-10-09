from mnist_classifier_interface import MnistClassifierInterface
import tensorflow as tf

class NnClassifier(MnistClassifierInterface):
    def __init__(self, model):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
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