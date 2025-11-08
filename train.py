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
# Reduce sequence length to produce more independent test windows and more training samples
max_seq_length = 200
num_features = 18
num_classes = 6
batch_size = 32

X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, scaler = prepare_pytorch_data(
    X_train, X_test, y_train, y_test, 
    max_seq_length=max_seq_length, 
    num_classes=num_classes
)

# Quick leakage check: compute min L2 distance from each test window to all train windows
try:
    X_train_np = X_train_tensor.numpy()
    X_test_np = X_test_tensor.numpy()
    min_dists = []
    for i in range(X_test_np.shape[0]):
        diffs = X_train_np - X_test_np[i:i+1]
        dists = np.linalg.norm(diffs.reshape(diffs.shape[0], -1), axis=1)
        min_dists.append(dists.min())
    min_dists = np.array(min_dists)
    print(f"\n[LEAKAGE CHECK] test windows: {len(min_dists)}, min/mean/max distance to closest train window: {min_dists.min():.6e}/{min_dists.mean():.6e}/{min_dists.max():.6e}")
    very_close = (min_dists < 1e-6).sum()
    print(f"[LEAKAGE CHECK] test windows with almost-identical train window (<1e-6): {very_close}")
except Exception as e:
    print(f"[LEAKAGE CHECK] skipped due to error: {e}")

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
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Training on device: {device}")


def debug_check_first_batch(model, train_loader, criterion, device):
    """Run one batch through the model and print diagnostics to catch NaNs/inf early."""
    model.eval()
    try:
        data, target = next(iter(train_loader))
    except StopIteration:
        print("No data in train_loader to debug.")
        return

    data, target = data.to(device), target.to(device)

    # Input checks
    print("\n[DEBUG] First batch stats:")
    print(f"  data dtype: {data.dtype}, shape: {data.shape}")
    print(f"  any NaN in input: {torch.isnan(data).any().item()}")
    print(f"  any Inf in input: {torch.isinf(data).any().item()}")
    try:
        print(f"  input min/max/mean: {data.min().item():.6f} / {data.max().item():.6f} / {data.mean().item():.6f}")
    except Exception:
        pass

    # Forward pass checks
    with torch.no_grad():
        output = model(data)
        print(f"  output shape: {output.shape}")
        print(f"  any NaN in output: {torch.isnan(output).any().item()}")
        print(f"  any Inf in output: {torch.isinf(output).any().item()}")
        try:
            print(f"  output min/max/mean: {output.min().item():.6f} / {output.max().item():.6f} / {output.mean().item():.6f}")
        except Exception:
            pass

    # Loss check
    loss = criterion(output, target)
    print(f"  loss on first batch: {loss.item()}")
    if torch.isnan(loss) or torch.isinf(loss):
        raise RuntimeError("NaN/Inf loss detected on first batch. Check inputs, model, or loss settings.")

    model.train()


def diagnose_batch(model, data, device):
    """Run a detailed forward pass and print intermediate stats for each branch to find where NaNs originate."""
    model.eval()
    data = data.to(device)
    batch_size = data.size(0)

    # Print input diagnostics
    try:
        print(f"[DIAG] input any NaN: {torch.isnan(data).any().item()} | any Inf: {torch.isinf(data).any().item()}")
        print(f"[DIAG] input min/max/mean: {data.min().item():.6f}/{data.max().item():.6f}/{data.mean().item():.6f}")
    except Exception:
        pass

    try:
        # CNN branch
        cnn_input = data.transpose(1, 2)
        cnn_out = torch.relu(model.cnn_conv1(cnn_input))
        print(f"[DIAG] CNN conv1 out NaN? {torch.isnan(cnn_out).any().item()} | min/max: {cnn_out.min().item():.6f}/{cnn_out.max().item():.6f}")
        cnn_out = model.cnn_pool1(cnn_out)
        cnn_out = torch.relu(model.cnn_conv2(cnn_out))
        print(f"[DIAG] CNN conv2 out NaN? {torch.isnan(cnn_out).any().item()} | min/max: {cnn_out.min().item():.6f}/{cnn_out.max().item():.6f}")
        cnn_out = model.cnn_pool2(cnn_out)
        cnn_out_flat = cnn_out.view(batch_size, -1)
        print(f"[DIAG] CNN flat NaN? {torch.isnan(cnn_out_flat).any().item()} | shape: {cnn_out_flat.shape}")

        # LSTM branch
        lstm_out1, (h1, c1) = model.lstm1(data)
        print(f"[DIAG] LSTM1 out NaN? {torch.isnan(lstm_out1).any().item()} | h1 min/max: {h1.min().item():.6f}/{h1.max().item():.6f}")
        lstm_out2, (h2, c2) = model.lstm2(lstm_out1)
        lstm_feat = h2[-1]
        print(f"[DIAG] LSTM2 feat NaN? {torch.isnan(lstm_feat).any().item()} | shape: {lstm_feat.shape}")

        # Transformer branch
        trans_out = model.transformer_block(data)
        print(f"[DIAG] Transformer block out NaN? {torch.isnan(trans_out).any().item()} | min/max: {trans_out.min().item():.6f}/{trans_out.max().item():.6f}")
        trans_feat = torch.mean(trans_out, dim=1)
        print(f"[DIAG] Transformer feat NaN? {torch.isnan(trans_feat).any().item()} | shape: {trans_feat.shape}")

        # Fusion
        concatenated = torch.cat([cnn_out_flat, lstm_feat, trans_feat], dim=1)
        print(f"[DIAG] Concatenated NaN? {torch.isnan(concatenated).any().item()} | shape: {concatenated.shape}")
        fused = model.fusion(concatenated)
        print(f"[DIAG] Fused NaN? {torch.isnan(fused).any().item()} | min/max: {fused.min().item():.6f}/{fused.max().item():.6f}")

        # Classifier
        out = model.classifier(fused)
        print(f"[DIAG] Final output NaN? {torch.isnan(out).any().item()} | min/max: {out.min().item():.6f}/{out.max().item():.6f}")
    except Exception as e:
        print(f"[DIAG ERROR] Exception during diagnosis: {e}")
    finally:
        model.train()

# Run quick debug check before training
try:
    debug_check_first_batch(model, train_loader, criterion, device)
except Exception as e:
    print(f"\n[ERROR] Debug check failed: {e}")
    print("Aborting training to avoid corrupting training run. Fix the root cause and retry.")
    sys.exit(1)

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

        # If loss is NaN or Inf, abort this epoch and surface diagnostics
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n[ERROR] Loss is NaN or Inf at batch {batch_idx+1}. Aborting training.\n")
            # Print some diagnostics
            print(f"  any NaN in output: {torch.isnan(output).any().item()}")
            print(f"  any Inf in output: {torch.isinf(output).any().item()}")
            # Run detailed diagnosis to find which branch produces NaNs
            try:
                diagnose_batch(model, data, device)
            except Exception as _:
                pass
            raise RuntimeError("NaN/Inf loss during training")

        loss.backward()

        # Gradient clipping to avoid explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Check for NaNs in gradients
        any_grad_nan = False
        for p in model.parameters():
            if p.grad is not None and torch.isnan(p.grad).any():
                any_grad_nan = True
                break
        if any_grad_nan:
            print(f"\n[ERROR] NaN detected in gradients at batch {batch_idx+1}. Aborting training.")
            raise RuntimeError("NaN in gradients")

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
