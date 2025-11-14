# predict.py

import pandas as pd
import numpy as np
import os
import re
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Import the PyTorch model definition
from model import HybridFaultPredictor

# --- Data Cleaning and Preprocessing Function ---

def load_and_preprocess_data(file_path):
    """
    Loads, cleans, and prepares the power line fault data.
    Handles complex number strings and imputes missing values.
    """
    print("--- 1. Loading and Cleaning Data ---")
    
    # 1. Load the dataset
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"ERROR: File not found at '{file_path}'. Please check the path.")
        return None, None, None, None, None
        
    # 2. Handle Complex Number Strings (ReactivePower columns)
    complex_cols = [col for col in df.columns if 'ReactivePower' in col]

    def extract_real_part_robust(value):
        """Robustly extracts the real part from a complex number string like '1234.56+0i'."""
        if isinstance(value, str):
            # Specific case for the observed format: 'real+0i'
            if '+0i' in value:
                try:
                    return float(value.replace('+0i', ''))
                except ValueError:
                    return np.nan
            
        try:
            return float(value)
        except:
            return np.nan

    for col in complex_cols:
        df[col] = df[col].apply(extract_real_part_robust)
    
    # Impute NaNs (found in ReactivePower columns) with the column median
    for col in complex_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)

    # 3. Separate Features (X) and Target (y)
    X = df.drop(columns=['FaultType'])
    y = df['FaultType']
    
    # 4. Target Encoding
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)
    print(f"Target Classes ({num_classes}): {le.classes_}")
    
    # 5. Feature Scaling and Reshaping
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    # Reshape data to (samples, time_steps=1, features) for the hybrid model
    time_steps = 1
    X_reshaped = X_scaled.reshape(-1, time_steps, X_scaled.shape[1]).astype(np.float32)
    
    print(f"Data Reshaped: X_reshaped.shape={X_reshaped.shape}, y_encoded.shape={y_encoded.shape}")
    print("Data cleaning and preprocessing complete.")
    
    return X_reshaped, y_encoded, num_classes, le


# --- PyTorch Dataset Class ---
class PowerLineDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        # Use LongTensor for integer labels (required by CrossEntropyLoss)
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- Main Execution ---
if __name__ == '__main__':
    
    # Set the path to your dataset file
    FILE_PATH = 'Datasets/NewDataSet-Comp.csv' 
    
    # --- 0. Device Configuration (CUDA Check) ---
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n--- 0. Device Configuration ---")
    if DEVICE.type == 'cuda':
        print(f"✅ Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ CUDA not available. Training on CPU. Please check PyTorch installation.")
        
    # --- 1. Data Preparation ---
    X_reshaped, y_encoded, num_classes, label_encoder = load_and_preprocess_data(FILE_PATH)
    
    if X_reshaped is not None:
        # Split Data (80% Train, 20% Test)
        X_train, X_test, y_train, y_test = train_test_split(
            X_reshaped, y_encoded, test_size=0.65, random_state=60, stratify=y_encoded
        )
        
        # Create PyTorch DataLoaders
        train_dataset = PowerLineDataset(X_train, y_train)
        test_dataset = PowerLineDataset(X_test, y_test)
        
        BATCH_SIZE = 64
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # --- 2. Model, Loss, and Optimizer ---
        input_size = X_train.shape[2]
        model = HybridFaultPredictor(input_size, num_classes).to(DEVICE)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # --- 3. Model Training ---
        NUM_EPOCHS = 30 
        print(f"\n--- 3. Model Training (on {DEVICE}) ---")
        
        best_acc = 0.0
        best_epoch = 0
        best_path = 'best_hybrid_fault_predictor_model.pth'
        train_losses = []
        train_accs = []
        test_accs = []
        
        for epoch in range(NUM_EPOCHS):
            model.train()
            total_loss = 0
            train_correct = 0
            train_total = 0
            
            # Use tqdm for a nice progress bar
            train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
            for inputs, labels in train_loop:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, train_pred = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (train_pred == labels).sum().item()
                train_loop.set_postfix(loss=total_loss / (train_loop.n + 1))

            # --- 4. Model Evaluation ---
            model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            train_accuracy = 100 * train_correct / train_total
            avg_loss = total_loss / len(train_loader)
            
            train_losses.append(avg_loss)
            train_accs.append(train_accuracy)
            test_accs.append(accuracy)
            
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}: Train Acc: {train_accuracy:.2f}% | Test Acc: {accuracy:.2f}% | Loss: {avg_loss:.4f}")
            
            # Save best model
            if accuracy > best_acc:
                best_acc = accuracy
                best_epoch = epoch + 1
                torch.save(model.state_dict(), best_path)
                print(f"  🔖 New best model saved (Test Acc={best_acc:.2f}%) -> {best_path}")
            
        # --- 5. Save Final Model ---
        model_save_path = 'hybrid_fault_predictor_model.pth'
        torch.save(model.state_dict(), model_save_path)
        print(f"\n--- 5. Model Saved ---")
        print(f"Final model saved to '{model_save_path}'")
        print(f"Best model saved to '{best_path}' (Epoch {best_epoch}, Test Acc={best_acc:.2f}%)")
        
        # --- 6. Plot Training History ---
        print(f"\n--- 6. Generating Training History Plot ---")
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(range(1, NUM_EPOCHS + 1), train_accs, label='Train Accuracy', marker='o')
        plt.plot(range(1, NUM_EPOCHS + 1), test_accs, label='Test Accuracy', marker='s')
        plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch ({best_epoch})')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Model Accuracy Over Epochs')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label='Train Loss', marker='o', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150)
        print("Training history plot saved as 'training_history.png'")
        
        # --- 7. Generate Confusion Matrix ---
        print(f"\n--- 7. Generating Confusion Matrix ---")
        model.load_state_dict(torch.load(best_path))
        model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        cm = confusion_matrix(all_labels, all_preds)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=label_encoder.classes_, 
                    yticklabels=label_encoder.classes_)
        plt.title(f'Confusion Matrix (Best Model - Epoch {best_epoch})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=150)
        print("Confusion matrix plot saved as 'confusion_matrix.png'")
        
        # --- 8. Final Summary ---
        print(f"\n{'='*60}")
        print(f"TRAINING SUMMARY")
        print(f"{'='*60}")
        print(f"Device used: {DEVICE}")
        print(f"Total epochs: {NUM_EPOCHS}")
        print(f"Best epoch: {best_epoch}")
        print(f"Best test accuracy: {best_acc:.2f}%")
        print(f"Final test accuracy: {test_accs[-1]:.2f}%")
        print(f"Best model saved as: '{best_path}'")
        print(f"Final model saved as: '{model_save_path}'")
        print(f"Training history plot: 'training_history.png'")
        print(f"Confusion matrix plot: 'confusion_matrix.png'")
        print(f"{'='*60}")
        
        print(f"\n--- Classification Report (Best Model) ---")
        print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))