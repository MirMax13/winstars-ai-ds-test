from task1_mnist_classifier.mnist_classifier_interface import MnistClassifierInterface

class NnClassifier(MnistClassifierInterface):
    def __init__(self, model):
        self.model = model

    def train(self, train_data, train_labels):
        pass

    def predict(self, test_data):
        pass