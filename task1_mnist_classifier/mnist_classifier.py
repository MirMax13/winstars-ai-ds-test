from task1_mnist_classifier.models.cnn_classifier import CnnClassifier
from task1_mnist_classifier.models.nn_classifier import NnClassifier
from task1_mnist_classifier.models.random_forest_classifier import RandomForestClassifier
from task1_mnist_classifier.mnist_classifier_interface import MnistClassifierInterface

class MnistClassifier(MnistClassifierInterface):
    def __init__(self, algorithm, model):
        if algorithm == 'cnn':
            self.classifier = CnnClassifier(model)
        elif algorithm == 'rf':
            self.classifier = RandomForestClassifier(model)
        elif algorithm == 'nn':
            self.classifier = NnClassifier(model)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def train(self, train_data, train_labels):
        self.classifier.train(train_data, train_labels)

    def predict(self, test_data):
        return self.classifier.predict(test_data)