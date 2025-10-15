import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter("ignore")


class ConvertToRGB:
    """Custom transform to ensure all images are in RGB format."""
    def __call__(self, img):
        if img.mode != "RGB":
            return img.convert("RGB")
        return img


def calculate_dataset_stats(data_dir, batch_size=32):
    print("Calculating dataset statistics...")
    
    transform_basic = transforms.Compose([
        ConvertToRGB(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    dataset = datasets.ImageFolder(data_dir, transform_basic)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    channel_sum, channel_sqr_sum, num_batches = 0, 0, 0
    
    for data, _ in tqdm(data_loader, desc="Computing mean and std"):
        channel_sum += torch.mean(data, dim=[0, 2, 3])
        channel_sqr_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1
    
    mean = channel_sum / num_batches
    std = (channel_sqr_sum / num_batches - mean ** 2) ** 0.5
    
    print(f"Dataset stats - Mean: {mean}, Std: {std}")
    return mean, std


def get_data_loaders(data_dir, batch_size=32, train_split=0.8, seed=42):
    # Calculate dataset statistics
    mean, std = calculate_dataset_stats(data_dir, batch_size)
    
    # Define augmentation and normalization transforms
    transform_norm = transforms.Compose([
        ConvertToRGB(),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(45),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # Load dataset
    print(f"Loading dataset from {data_dir}...")
    norm_dataset = datasets.ImageFolder(data_dir, transform_norm)
    class_names = norm_dataset.classes
    print(f"Found {len(norm_dataset)} images across {len(class_names)} classes")
    print(f"Classes: {class_names}")
    
    # Split dataset
    g = torch.Generator().manual_seed(seed)
    train_size = int(train_split * len(norm_dataset))
    val_size = len(norm_dataset) - train_size
    train_dataset, val_dataset = random_split(norm_dataset, [train_size, val_size], generator=g)
    
    print(f"Train set: {len(train_dataset)} images, Val set: {len(val_dataset)} images")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, class_names, mean, std


def create_resnet_model(num_classes=10, device='cuda'):
    print("Building ResNet50 model...")
    
    # Load pre-trained ResNet50
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    
    # Freeze all base layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace classifier head
    in_features = model.fc.in_features
    classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )
    model.fc = classifier
    
    model = model.to(device)
    print(f"Model created with {num_classes} output classes")
    
    return model


def train_epoch(model, loader, loss_fn, optimizer, device):
    """Train model for one epoch."""
    train_loss = 0
    model.train()
    
    for data, label in tqdm(loader, desc="Training"):
        optimizer.zero_grad()
        data = data.to(device)
        label = label.to(device)
        
        output = model(data)
        loss = loss_fn(output, label)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * data.size(0)
    
    return train_loss / len(loader.dataset)


def validate_epoch(model, loader, loss_fn, device):
    """Validate model on validation set."""
    total_loss = 0
    total_correct = 0
    model.eval()
    
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Validating"):
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            
            total_loss += loss.item() * imgs.size(0)
            correct = torch.eq(torch.argmax(outputs, dim=1), labels)
            total_correct += torch.sum(correct).item()
    
    avg_loss = total_loss / len(loader.dataset)
    accuracy = total_correct / len(loader.dataset)
    
    return avg_loss, accuracy


def train_model(model, train_loader, val_loader, loss_fn, optimizer, device, 
                scheduler=None, epochs=20, early_stop=5, checkpoint="resnet_model.pth"):
    
    print(f"Starting training for {epochs} epochs...")
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    lr_history = []
    
    best_val_loss = float('inf')
    early_stop_counter = 0
    
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        
        # Training
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        
        # Validation
        val_loss, val_acc = validate_epoch(model, val_loader, loss_fn, device)
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # Log learning rate
        lr = optimizer.param_groups[0]['lr']
        lr_history.append(lr)
        
        # Scheduler step
        if scheduler:
            scheduler.step()
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc*100:.2f}%, LR: {lr:.6f}")
        
        # Early stopping and checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
            }, checkpoint)
            
            print(f"Checkpoint saved (Val Loss: {val_loss:.4f})")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"No improvement ({early_stop_counter}/{early_stop})")
        
        # Check early stopping
        if early_stop_counter >= early_stop:
            print(f"Early stopping triggered after {epoch} epochs")
            break
    
    print(f"Training completed. Best validation loss: {best_val_loss:.4f}")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'lr_history': lr_history,
        'final_epoch': epoch,
        'best_val_loss': best_val_loss
    }


def plot_training_history(history, save_path='training_history.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_losses']) + 1)
    
    # Loss plot
    ax1.plot(epochs, history['train_losses'], 'b-o', label='Train Loss')
    ax1.plot(epochs, history['val_losses'], 'r-o', label='Val Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(epochs, [acc * 100 for acc in history['val_accuracies']], 'g-o')
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train Animal Image Classification Model")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--checkpoint", type=str, default="resnet_model.pth", help="Path to save model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Fixed parameters
    num_classes = 10
    train_split = 0.8
    early_stop = 10
    lr = 0.001
    step_size = 5
    gamma = 0.1
    
    seed = args.seed
    
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Get data loaders
    train_loader, val_loader, class_names, mean, std = get_data_loaders(
        args.data_dir, 
        batch_size=args.batch_size, 
        train_split=train_split,
        seed=seed
    )
    
    # Create model
    model = create_resnet_model(num_classes=num_classes, device=device)
    
    # Loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    # Train model
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        epochs=args.epochs,
        early_stop=early_stop,
        checkpoint=args.checkpoint
    )
    
    # Plot training history
    plot_path = args.checkpoint.replace('.pth', '_history.png')
    plot_training_history(history, save_path=plot_path)
    
    # Save dataset statistics for inference
    stats_path = args.checkpoint.replace('.pth', '_stats.pth')
    torch.save({
        'mean': mean,
        'std': std,
        'class_names': class_names
    }, stats_path)
    print(f"Dataset statistics saved to {stats_path}")
    
    print(f"Training completed successfully!")
    print(f"Model checkpoint: {args.checkpoint}")
    print(f"Training plot: {plot_path}")


if __name__ == "__main__":
    main()