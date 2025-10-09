from mnist_classifier_interface import MnistClassifierInterface
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier

class RandomForestClassifier(MnistClassifierInterface):
    def __init__(self, n_estimators=100, random_state=42):
        self.model = SklearnRandomForestClassifier(
            n_estimators=n_estimators, 
            random_state=random_state
        )

    def train(self, train_data, train_labels):
        self.model.fit(train_data, train_labels)

    def predict(self, test_data):
        return self.model.predict(test_data)