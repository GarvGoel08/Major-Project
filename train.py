import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from main import build_hybrid_model, device
from data_loader import load_datasets, prepare_pytorch_data, create_data_loaders
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import time
import sys

# Load and prepare the data
print("Loading datasets...")
X_train, X_test, y_train, y_test, class_to_idx = load_datasets('Datasets/Train')

print("\nPreparing data for PyTorch model...")
max_seq_length = 500
num_features = 18
num_classes = 6
batch_size = 32

X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, scaler = prepare_pytorch_data(
    X_train, X_test, y_train, y_test, 
    max_seq_length=max_seq_length, 
    num_classes=num_classes
)

# Create data loaders
train_loader, test_loader = create_data_loaders(
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, batch_size=batch_size
)

# Build the model
print("\nBuilding model...")
embed_dim = num_features
num_heads = 2
ff_dim = 32

model = build_hybrid_model(
    input_shape=(max_seq_length, num_features),
    num_classes=num_classes,
    max_seq_length=max_seq_length,
    embed_dim=embed_dim,
    num_heads=num_heads,
    ff_dim=ff_dim
)

# Setup optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Training on device: {device}")

# Training function with progress bar
def train_epoch(model, train_loader, optimizer, criterion, device, epoch, epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # Progress tracking
    num_batches = len(train_loader)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        # Calculate current metrics
        current_loss = total_loss / (batch_idx + 1)
        current_acc = 100. * correct / total
        
        # Create progress bar
        progress = (batch_idx + 1) / num_batches
        bar_length = 30
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        # Print progress (overwrite same line)
        print(f'\rEpoch {epoch+1}/{epochs} [{bar}] {batch_idx+1}/{num_batches} - '
              f'loss: {current_loss:.4f} - acc: {current_acc:.2f}%', end='', flush=True)
    
    print()  # New line after progress bar
    return total_loss / len(train_loader), 100. * correct / total

# Validation function
def validate_epoch(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    return test_loss / len(test_loader), 100. * correct / total

# Train the model
print("\nTraining model...")
epochs = 50
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

start_time = time.time()
best_accuracy = 0

for epoch in range(epochs):
    epoch_start = time.time()
    
    # Training with progress bar
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, epoch, epochs)
    
    # Validation (quick, no progress bar needed)
    val_loss, val_acc = validate_epoch(model, test_loader, criterion, device)
    
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    
    epoch_time = time.time() - epoch_start
    
    # Print validation results and summary on same line as progress bar ended
    print(f'val_loss: {val_loss:.4f} - val_acc: {val_acc:.2f}% - {epoch_time:.1f}s', end='')
    
    # Save best model
    if val_acc > best_accuracy:
        best_accuracy = val_acc
        torch.save(model.state_dict(), 'best_fault_detection_model.pth')
        print(f' - 💾 New best!', end='')
    
    print()  # Final newline for this epoch

total_time = time.time() - start_time
print(f"Training completed in {total_time:.1f} seconds")
print(f"Best validation accuracy: {best_accuracy:.2f}%")

# Save final model
torch.save(model.state_dict(), 'fault_detection_model.pth')
print("\nFinal model saved as 'fault_detection_model.pth'")

# Evaluate the model
print("\nEvaluating model on test set...")
final_loss, final_accuracy = validate_epoch(model, test_loader, criterion, device)
print(f"Final Test Loss: {final_loss:.4f}")
print(f"Final Test Accuracy: {final_accuracy:.2f}%")

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.png')
print("Training history plot saved as 'training_history.png'")

# Get predictions for detailed evaluation
print("\nGenerating predictions for detailed evaluation...")
model.eval()
all_predictions = []
all_targets = []

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        predictions = output.argmax(dim=1)
        
        all_predictions.extend(predictions.cpu().numpy())
        all_targets.extend(target.cpu().numpy())

y_pred_classes = np.array(all_predictions)
y_test_classes = np.array(all_targets)

# Inverse mapping
idx_to_class = {v: k for k, v in class_to_idx.items()}
class_names = [idx_to_class[i] for i in range(num_classes)]

print("\nClassification Report:")
print(classification_report(y_test_classes, y_pred_classes, target_names=class_names))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test_classes, y_pred_classes)
print(cm)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)

# Add text annotations
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("Confusion matrix plot saved as 'confusion_matrix.png'")

print(f"\n{'='*50}")
print("TRAINING SUMMARY")
print(f"{'='*50}")
print(f"Device used: {device}")
print(f"Total training time: {total_time:.1f} seconds")
print(f"Best validation accuracy: {best_accuracy:.2f}%")
print(f"Final test accuracy: {final_accuracy:.2f}%")
print(f"Model saved as: 'fault_detection_model.pth'")
print(f"Best model saved as: 'best_fault_detection_model.pth'")
if torch.cuda.is_available():
    print(f"GPU memory used: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
