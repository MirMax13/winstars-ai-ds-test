"""
Random Forest Classifier for MNIST

This module implements a Random Forest classifier for digit recognition.
It uses scikit-learn's RandomForestClassifier with optimized parameters.
"""

from mnist_classifier_interface import MnistClassifierInterface
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier


class RandomForestClassifier(MnistClassifierInterface):
    """
    Random Forest implementation for MNIST digit classification.
    
    This classifier uses an ensemble of decision trees to classify handwritten digits.
    It's fast to train and provides good baseline performance.
    
    Attributes:
        model: Underlying scikit-learn RandomForestClassifier
    """
    
    def __init__(self, n_estimators=100, random_state=42):
        """
        Initialize the Random Forest classifier.
        
        Args:
            n_estimators (int): Number of trees in the forest. Default: 100
            random_state (int): Random seed for reproducibility. Default: 42
        """
        self.model = SklearnRandomForestClassifier(
            n_estimators=n_estimators, 
            random_state=random_state
        )

    def train(self, train_data, train_labels):
        """
        Train the Random Forest on MNIST data.
        
        Args:
            train_data: Flattened training images (shape: [n_samples, 784])
            train_labels: Training labels (0-9 digits)
        """
        self.model.fit(train_data, train_labels)

    def predict(self, test_data):
        """
        Predict digits for test images.
        
        Args:
            test_data: Flattened test images (shape: [n_samples, 784])
            
        Returns:
            Predicted labels as numpy array
        """
        return self.model.predict(test_data)
