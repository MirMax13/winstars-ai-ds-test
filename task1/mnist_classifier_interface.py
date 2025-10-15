"""
MNIST Classifier Interface

This module defines the abstract base class that all MNIST classifiers must implement.
It ensures a consistent API across different classification algorithms (Random Forest, NN, CNN).
"""


class MnistClassifierInterface:
    """
    Abstract interface for MNIST digit classifiers.
    
    All classifier implementations must inherit from this class and implement
    the train() and predict() methods to ensure a uniform interface.
    """
    
    def train(self, train_data, train_labels):
        """
        Train the classifier on MNIST data.
        
        Args:
            train_data: Training images (format depends on the specific classifier)
            train_labels: Corresponding labels (0-9 digits)
        """
        pass

    def predict(self, test_data):
        """
        Make predictions on test data.
        
        Args:
            test_data: Test images (format depends on the specific classifier)
            
        Returns:
            Predicted labels as a numpy array
        """
        pass
