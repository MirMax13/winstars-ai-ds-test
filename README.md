# WinStars AI/DS Internship Test

This repository contains solutions for the WinStars AI/DS internship test assignment.

## Project Structure

```
winstars-ai-ds-test/
├── task1/          # MNIST digit classification with OOP
├── task2/          # Animal detection pipeline (NER + Image Classification)
└── README.md       # This file
```

## Task 1: MNIST Classification with OOP

Implementation of three classification algorithms (Random Forest, Neural Network, CNN) with a unified interface for MNIST digit recognition.

**Key Features:**
- Abstract interface for all classifiers
- Factory pattern for algorithm selection
- Three model implementations: RF, NN, CNN
- Demo notebook with performance comparison

[Full documentation →](task1/README.md)

## Task 2: Animal Detection Pipeline

ML pipeline combining NER and image classification to verify text statements about animals in images.

**Key Features:**
- DistilBERT-based NER for animal entity extraction
- ResNet50 transfer learning for image classification
- Automated negation handling
- End-to-end verification pipeline

[Full documentation →](task2/README.md)

## Quick Start

Each task is self-contained with its own setup instructions:

```powershell
# Task 1
cd task1
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Task 2
cd task2
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- See individual `requirements.txt` files for specific dependencies

## Author

Maxym Matskiv

## License

This project is part of a test assignment for WinStars AI/DS position.
