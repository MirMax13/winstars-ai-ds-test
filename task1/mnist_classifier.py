from models.cnn_classifier import CnnClassifier
from models.nn_classifier import NnClassifier
from models.random_forest_classifier import RandomForestClassifier
from mnist_classifier_interface import MnistClassifierInterface

class MnistClassifier(MnistClassifierInterface):
    def __init__(self, algorithm):
        if algorithm == "rf":
            self.classifier = RandomForestClassifier()
        elif algorithm == "nn":
            self.classifier = NnClassifier(None)
        elif algorithm == "cnn":
            self.classifier = CnnClassifier(None)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def train(self, train_data, train_labels):
        self.classifier.train(train_data, train_labels)

    def predict(self, test_data):
        return self.classifier.predict(test_data)