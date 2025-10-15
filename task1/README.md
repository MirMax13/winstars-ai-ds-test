# Task 1: MNIST Digit Classification

A simple implementation of handwritten digit classification using the MNIST dataset. This solution provides three different approaches with a unified interface for easy comparison.

## Overview

The project implements three classification methods:
- **Random Forest** - Traditional machine learning approach
- **Neural Network** - Simple feedforward network
- **CNN** - Convolutional Neural Network

All models follow the same interface defined in `mnist_classifier_interface.py`, making it easy to switch between different approaches.

## Project Structure

```
task1/
├── mnist_classifier.py              # Main classifier with unified interface
├── mnist_classifier_interface.py    # Abstract base class
├── models/
│   ├── random_forest_classifier.py  # Random Forest implementation
│   ├── nn_classifier.py             # Neural Network implementation
│   └── cnn_classifier.py            # CNN implementation
├── notebook/
│   └── demo_1.ipynb                 # Demo notebook with examples
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Setup

### Prerequisites
- Python 3.8 or higher

### Installation

1. **Clone the repository** (if you haven't already):
```bash
git clone https://github.com/MirMax13/winstars-ai-ds-test.git
cd winstars-ai-ds-test/task1
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
```

3. **Activate the virtual environment**:
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - Linux/Mac:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Example

```python
from mnist_classifier import MnistClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Load MNIST data
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
y = y.astype('int')

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Random Forest
rf_clf = MnistClassifier("rf")
rf_clf.train(X_train, y_train)
predictions_rf = rf_clf.predict(X_test)

# Neural Network
nn_clf = MnistClassifier("nn")
nn_clf.train(X_train.reshape(-1, 28, 28), y_train)
predictions_nn = nn_clf.predict(X_test.reshape(-1, 28, 28))

# CNN
cnn_clf = MnistClassifier("cnn")
cnn_clf.train(X_train.reshape(-1, 28, 28, 1), y_train)
predictions_cnn = cnn_clf.predict(X_test.reshape(-1, 28, 28, 1))
```

### Running the Demo

To see a complete demonstration with visualizations:

```bash
jupyter notebook notebook/demo_1.ipynb
```

The demo notebook includes:
- Data loading and visualization
- Training all three models
- Performance comparison
- Edge case analysis
- Visual results

## Model Details

### Random Forest Classifier
- **Input**: Flattened 784-dimensional vector (28×28 pixels)
- **Parameters**: 100 trees, max depth 20
- **Pros**: Fast training, good baseline
- **Cons**: Lower accuracy on complex patterns

### Neural Network
- **Input**: 28×28 image
- **Architecture**: 
  - Flatten → Dense(128, ReLU) → Dropout(0.2) → Dense(10, Softmax)
- **Training**: 10 epochs, batch size 32
- **Pros**: Better feature learning than RF
- **Cons**: Longer training time

### CNN (Convolutional Neural Network)
- **Input**: 28×28×1 image
- **Architecture**:
  - Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Flatten → Dense(128) → Dense(10)
- **Training**: 10 epochs, batch size 32
- **Pros**: Best accuracy, learns spatial features
- **Cons**: Slowest training

## Performance

On a subset of 10,000 MNIST samples:
- **Random Forest**: ~95-96% accuracy, fastest training
- **Neural Network**: ~97-98% accuracy, moderate training time
- **CNN**: ~98-99% accuracy, slowest training

*Note: Actual performance may vary depending on dataset size and hardware.*

## Edge Cases

The models may struggle with:
- Poorly written or ambiguous digits
- Rotated or skewed digits
- Digits with unusual stroke patterns
- Similar-looking digits (e.g., 4 vs 9, 3 vs 8)

See the demo notebook for visual examples of challenging cases.

## Requirements

Main dependencies (see `requirements.txt` for full list):
- `numpy` - Numerical computations
- `scikit-learn` - Random Forest and data utilities
- `tensorflow` / `keras` - Neural networks
- `matplotlib` - Visualization
- `jupyter` - Interactive notebooks

## License

This project is part of a test assignment for WinStars AI/DS position.

## Author

Maxym Matskiv
