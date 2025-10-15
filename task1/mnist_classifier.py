"""
MNIST Classifier Facade

This module provides a unified interface to all MNIST classifiers.
It acts as a factory that creates the appropriate classifier based on the algorithm parameter.
"""

from models.cnn_classifier import CnnClassifier
from models.nn_classifier import NnClassifier
from models.random_forest_classifier import RandomForestClassifier
from mnist_classifier_interface import MnistClassifierInterface


class MnistClassifier(MnistClassifierInterface):
    """
    Unified MNIST classifier that delegates to specific algorithm implementations.
    
    This class provides a single entry point for training and prediction,
    regardless of the underlying algorithm (Random Forest, Neural Network, or CNN).
    
    Usage:
        classifier = MnistClassifier("rf")  # for Random Forest
        classifier = MnistClassifier("nn")  # for Neural Network
        classifier = MnistClassifier("cnn") # for CNN
    """
    
    def __init__(self, algorithm):
        """
        Initialize the classifier with the specified algorithm.
        
        Args:
            algorithm (str): Algorithm type - one of "rf", "nn", or "cnn"
            
        Raises:
            ValueError: If algorithm is not one of the supported types
        """
        if algorithm == "rf":
            self.classifier = RandomForestClassifier()
        elif algorithm == "nn":
            self.classifier = NnClassifier(None)
        elif algorithm == "cnn":
            self.classifier = CnnClassifier(None)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}. Choose from: 'rf', 'nn', 'cnn'")

    def train(self, train_data, train_labels):
        """
        Train the underlying classifier.
        
        Args:
            train_data: Training images (format depends on algorithm)
            train_labels: Corresponding labels (0-9 digits)
        """
        self.classifier.train(train_data, train_labels)

    def predict(self, test_data):
        """
        Make predictions using the trained classifier.
        
        Args:
            test_data: Test images (format depends on algorithm)
            
        Returns:
            Predicted labels as a numpy array
        """
        return self.classifier.predict(test_data)
