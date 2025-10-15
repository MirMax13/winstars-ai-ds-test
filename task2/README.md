# Task 2: Animal Classification - NER + Image Classification Pipeline

This project implements a complete pipeline for animal detection and classification combining:
- **Named Entity Recognition (NER)** for extracting animal mentions from text
- **Image Classification** using ResNet50 transfer learning for visual animal recognition

## Project Structure

```
task2/
├── cnn/                          # Image classification module
│   ├── training.py              # PyTorch training script (ResNet50)
│   ├── inference_image.py       # Image inference script
│   └── animal_classification.py # Full training code
├── ner/                          # Named Entity Recognition module
│   ├── generate_ner_dataset.py  # Dataset generation
│   ├── train_ner.py             # NER model training (DistilBERT)
│   └── inference_ner.py         # Text inference script
├── pipeline/
│   └── pipeline.py              # Unified NER + Image verification
├── notebooks/
│   ├── dataset_eda.ipynb        # Exploratory Data Analysis
│   └── demo_2.ipynb             # Solution demonstration
├── raw-img/                      # Dataset directory
├── requirements.txt              # PyTorch dependencies
└── README.md                     # This file
```

## Setup

### 1. Setup Environment

```powershell
# Create virtual environment
python -m venv venv

# Activate environment
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Train Image Classification Model

```powershell
cd cnn

# Basic training
python training.py --data_dir ./raw-img

# With custom parameters
python training.py --data_dir ./raw-img --epochs 50 --batch_size 16
```

**Available Arguments:**
- `--data_dir`: Path to dataset directory (required)
- `--epochs`: Number of epochs (default: 100)
- `--batch_size`: Batch size (default: 32)
- `--checkpoint`: Output model path (default: resnet_model.pth)

**Fixed Settings:**
- Device: Auto-detected (GPU if available)
- Number of classes: 10
- Early stopping patience: 10 epochs
- Learning rate: 0.001
- LR scheduler: StepLR (step_size=5, gamma=0.1)

### 3. Run Image Inference

```powershell
# Single image prediction
python inference_image.py --model_path resnet_model.pth --image_path test_dog.jpg

# Save results to JSON
python inference_image.py --model_path resnet_model.pth --image_path test_dog.jpg --output results.json
```

**Inference Arguments:**
- `--model_path`: Path to trained model checkpoint (required)
- `--image_path`: Path to input image (required)
- `--output`: Save results to JSON file (optional)

**Fixed Settings:**
- Device: Auto-detected (GPU if available)
- Number of classes: 10
- Top predictions shown: 3

### 4. Train NER Model

```powershell
cd ../ner

# Generate training dataset
python generate_ner_dataset.py

# Train NER model
python train_ner.py --output_dir ./ner_model --epochs 3
```

### 5. Run Complete Pipeline

```powershell
cd pipeline

# Verify statement with text and image
python pipeline.py
```

The pipeline combines NER and image classification to verify if a text statement matches the image content. It handles negations automatically.

**Example in code:**
```python
from pipeline import check_statement

text = "I see a dog"
image_path = "../cnn/raw-img/cane/1.jpeg"
result = check_statement(text, image_path)  # Returns True or False
```

## Model Architecture

### Image Classification (ResNet50)
```
ResNet50 (Pre-trained on ImageNet)
├── Frozen Base Layers
└── Custom Classifier:
    ├── Linear(2048 → 256)
    ├── ReLU()
    ├── Dropout(0.5)
    └── Linear(256 → 10)
```

**Classes:** butterfly, cat, chicken, cow, dog, elephant, horse, sheep, spider, squirrel

**Data Augmentation:**
- Random resized crop (224x224)
- Random horizontal flip
- Random rotation (±45°)
- Normalization (computed from dataset)

### NER Model
- **Base:** DistilBERT-base-uncased
- **Fine-tuned:** Animal entity recognition
- **Labels:** O, B-ANIMAL, I-ANIMAL
- **Dataset:** 370 examples (positive + negative templates)

## Performance

### Image Classification
- **Validation Accuracy:** ~95-98% (with transfer learning)
- **Training Time:** ~20-30 epochs (with early stopping)
- **Best Practices:**
  - Early stopping prevents overfitting
  - Learning rate scheduling improves convergence
  - Data augmentation enhances generalization

### NER Model
- **Accuracy:** High precision on animal mentions
- **Handles:** Negations and contextual variations

## Usage Examples

### Training with Custom Settings

```powershell
# Quick training (fewer epochs)
python cnn/training.py --data_dir ./cnn/raw-img --epochs 20

# Smaller batch size (for limited memory)
python cnn/training.py --data_dir ./cnn/raw-img --batch_size 16

# Custom model name
python cnn/training.py --data_dir ./cnn/raw-img --checkpoint best_model.pth
```

### Inference Examples

```powershell
# Basic prediction
python cnn/inference_image.py --model_path cnn/resnet_model.pth --image_path test.jpg

# Save to JSON
python cnn/inference_image.py --model_path cnn/resnet_model.pth --image_path test.jpg --output result.json
```

### Pipeline Integration

```python
from pipeline import verify_statement

# Verify animal statement
result = verify_statement(
    text="There's a beautiful butterfly",
    image_path="images/butterfly.jpg"
)
print(f"Statement is {result}")  # True or False
```

## Troubleshooting

### PyTorch Not Using GPU

```powershell
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# If False, install CUDA-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Out of Memory Errors

```powershell
# Reduce batch size
python training.py --data_dir ./raw-img --batch_size 16

# Use CPU (slower but more memory)
python training.py --data_dir ./raw-img --device cpu
```

### Import Errors

```powershell
# Reinstall all dependencies
pip install -r requirements.txt --upgrade
```

## Training Output

The training script generates:
- `resnet_model.pth` - Model checkpoint with best validation loss
- `resnet_model_stats.pth` - Dataset normalization statistics
- `training_history.png` - Loss and accuracy curves

## Visualization Notebooks

- **`notebooks/dataset_eda.ipynb`** - Dataset exploration with class distribution, image dimensions, color analysis
- **`notebooks/demo_2.ipynb`** - Solution demonstration with model evaluation and pipeline testing

## Notes

- **Normalization:** Mean and std are calculated from the dataset and saved with the model
- **Class Mapping:** Italian folder names are automatically mapped to English
- **Early Stopping:** Prevents overfitting by monitoring validation loss
- **Checkpointing:** Only the best model (lowest val loss) is saved

## Dependencies

See `requirements.txt` for full list. Key packages:
- PyTorch ≥2.0.0
- torchvision ≥0.15.0
- transformers ≥4.30.0 (for NER)
- scikit-learn ≥1.3.0
- matplotlib, seaborn (visualization)

## References

The ResNet50 transfer learning approach was inspired by: [Animal Classification on Kaggle](https://www.kaggle.com/code/hopesb/animal-classification)

## Author

Maxym Matskiv