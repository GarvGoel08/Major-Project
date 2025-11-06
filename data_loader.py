import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F

# GPU device detection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_datasets(data_dir='Datasets/Train'):
    """
    Load all Excel datasets from the specified directory.
    
    Returns:
        X_train, X_test, y_train, y_test: Split datasets
    """
    
    # Define the mapping of files to class labels
    file_label_mapping = {
        'NormalParameters.xlsx': 'Normal',
        'Overload.xlsx': 'Overload',
        'LLG.xlsx': 'LLG',
        'LL.xlsx': 'LL',
        'LLLG.xlsx': 'LLLG',
        'LG.xlsx': 'LG'
    }
    
    # Class to index mapping
    class_to_idx = {
        'Normal': 0,
        'Overload': 1,
        'LLG': 2,
        'LL': 3,
        'LLLG': 4,
        'LG': 5
    }
    
    # Expected column names (18 features)
    expected_columns = [
        'Voltage1', 'Voltage2', 'Voltage3',
        'Current1', 'Current2', 'Current3',
        'ActivePower1', 'ActivePower2', 'ActivePower3',
        'ReactivePower1', 'ReactivePower2', 'ReactivePower3',
        'VoltageTHD1', 'VoltageTHD2', 'VoltageTHD3',
        'CurrentTHD1', 'CurrentTHD2', 'CurrentTHD3'
    ]
    
    all_data = []
    all_labels = []
    
    # Load each dataset
    for filename, label in file_label_mapping.items():
        filepath = os.path.join(data_dir, filename)
        
        if os.path.exists(filepath):
            print(f"Loading {filename}...")
            df = pd.read_excel(filepath)
            
            # If column names don't match, assume they're in the correct order
            if list(df.columns) != expected_columns:
                print(f"  Warning: Column names don't match expected. Using first 18 columns.")
                df = df.iloc[:, 2:20]
                df.columns = expected_columns
            
            # Get the data
            data = df[expected_columns].values
            
            # Use 17000 entries for training, rest for testing
            n_samples = len(data)
            train_size = min(17000, n_samples)  # Use 17000 or less if dataset is smaller

            # Use first 17000 for training
            train_data = data[:train_size]
            # Use the rest for testing
            test_data = data[train_size:]
            print(train_data)
            # Create labels
            train_labels = np.full(len(train_data), class_to_idx[label])
            test_labels = np.full(len(test_data), class_to_idx[label])
            
            all_data.append({
                'train': train_data,
                'test': test_data,
                'train_labels': train_labels,
                'test_labels': test_labels
            })
            
            print(f"  Loaded {len(data)} samples ({len(train_data)} train, {len(test_data)} test) for class '{label}'")
        else:
            print(f"Warning: {filepath} not found!")
    
    # Combine all data
    X_train = np.vstack([d['train'] for d in all_data])
    X_test = np.vstack([d['test'] for d in all_data])
    y_train = np.hstack([d['train_labels'] for d in all_data])
    y_test = np.hstack([d['test_labels'] for d in all_data])
    
    print(f"\nTotal samples: Train={len(X_train)}, Test={len(X_test)}")
    print(f"Feature shape: {X_train.shape[1]}")
    print(f"Classes distribution in training set:")
    for class_name, idx in class_to_idx.items():
        count = np.sum(y_train == idx)
        print(f"  {class_name}: {count}")
    
    return X_train, X_test, y_train, y_test, class_to_idx


def prepare_sequences(X, y, max_seq_length=500):
    """
    Prepare data for the model by creating sequences.
    If data has fewer samples than max_seq_length, pad with zeros.
    If data has more samples, create overlapping windows.
    
    Args:
        X: Input features (n_samples, n_features)
        y: Labels (n_samples,)
        max_seq_length: Length of sequences for the model
    
    Returns:
        X_seq: Sequences (n_sequences, max_seq_length, n_features)
        y_seq: Labels for sequences (n_sequences,)
    """
    n_samples, n_features = X.shape
    
    if n_samples < max_seq_length:
        # Pad with zeros
        padding = np.zeros((max_seq_length - n_samples, n_features))
        X_padded = np.vstack([X, padding])
        X_seq = X_padded[np.newaxis, :, :]  # Add batch dimension
        y_seq = np.array([y[0]])  # Use the first label (all should be same)
    else:
        # Create overlapping windows
        step_size = max_seq_length // 2  # 50% overlap
        X_seq = []
        y_seq = []
        
        for i in range(0, n_samples - max_seq_length + 1, step_size):
            X_seq.append(X[i:i + max_seq_length])
            # Use the most common label in the window
            y_seq.append(y[i])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
    
    return X_seq, y_seq


def prepare_data_for_model(X_train, X_test, y_train, y_test, max_seq_length=100, num_classes=6):
    """
    Prepare and preprocess data for the model with proper windowing.
    
    Returns:
        X_train_seq, X_test_seq, y_train_seq, y_test_seq, scaler
    """
    
    # Normalize the data
    scaler = StandardScaler()
    # Fit scaler on training data first
    scaler.fit(X_train)

    # Protect against zero std (which would cause division by zero -> NaN)
    if hasattr(scaler, 'scale_'):
        zero_scale_mask = scaler.scale_ == 0
        if np.any(zero_scale_mask):
            print("  Warning: some features have zero standard deviation. Replacing zeros with 1.0 to avoid NaNs.")
            scaler.scale_[zero_scale_mask] = 1.0

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create proper sliding windows instead of repetition
    def create_windows(X, y, window_size):
        if len(X) < window_size:
            # If we have fewer samples than window size, pad with last values
            padding_needed = window_size - len(X)
            X_padded = np.vstack([X, np.tile(X[-1], (padding_needed, 1))])
            return X_padded[np.newaxis, :, :], np.array([y[0]])
        else:
            # Create overlapping windows
            windows = []
            labels = []
            step_size = max(1, window_size // 4)  # 75% overlap
            
            for i in range(0, len(X) - window_size + 1, step_size):
                windows.append(X[i:i + window_size])
                labels.append(y[i])  # Use label at start of window
            
            return np.array(windows), np.array(labels)
    
    # Process each class separately to maintain class information
    X_train_windows = []
    y_train_windows = []
    X_test_windows = []
    y_test_windows = []
    
    # Group by class
    for class_idx in range(num_classes):
        # Training data for this class
        train_mask = y_train == class_idx
        if np.any(train_mask):
            class_X_train = X_train_scaled[train_mask]
            class_y_train = y_train[train_mask]
            
            X_win, y_win = create_windows(class_X_train, class_y_train, max_seq_length)
            X_train_windows.append(X_win)
            y_train_windows.append(y_win)
        
        # Test data for this class
        test_mask = y_test == class_idx
        if np.any(test_mask):
            class_X_test = X_test_scaled[test_mask]
            class_y_test = y_test[test_mask]
            
            X_win, y_win = create_windows(class_X_test, class_y_test, max_seq_length)
            X_test_windows.append(X_win)
            y_test_windows.append(y_win)
    
    # Combine all windows
    X_train_seq = np.vstack(X_train_windows)
    y_train_seq = np.hstack(y_train_windows)
    X_test_seq = np.vstack(X_test_windows)
    y_test_seq = np.hstack(y_test_windows)

    # Check for NaNs introduced during preprocessing/windowing
    nan_train_windows = np.isnan(X_train_seq).any(axis=(1, 2))
    if np.any(nan_train_windows):
        print(f"  Warning: {np.sum(nan_train_windows)} training windows contain NaNs. Indices: {np.where(nan_train_windows)[0][:10]}")
        # Replace NaNs with zero (safe fallback) and continue
        X_train_seq[np.isnan(X_train_seq)] = 0.0

    nan_test_windows = np.isnan(X_test_seq).any(axis=(1, 2))
    if np.any(nan_test_windows):
        print(f"  Warning: {np.sum(nan_test_windows)} test windows contain NaNs. Indices: {np.where(nan_test_windows)[0][:10]}")
        X_test_seq[np.isnan(X_test_seq)] = 0.0
    
    # Shuffle training data
    train_indices = np.random.permutation(len(X_train_seq))
    X_train_seq = X_train_seq[train_indices]
    y_train_seq = y_train_seq[train_indices]
    
    print(f"\nImproved data shapes:")
    print(f"  X_train: {X_train_seq.shape}")
    print(f"  X_test: {X_test_seq.shape}")
    print(f"  y_train: {y_train_seq.shape}")
    print(f"  y_test: {y_test_seq.shape}")
    
    return X_train_seq, X_test_seq, y_train_seq, y_test_seq, scaler


def prepare_pytorch_data(X_train, X_test, y_train, y_test, max_seq_length=100, num_classes=6):
    """
    Prepare data specifically for PyTorch training with GPU support.
    
    Returns:
        DataLoaders and tensors ready for PyTorch training
    """
    # Get processed data with proper windowing
    X_train_seq, X_test_seq, y_train_seq, y_test_seq, scaler = prepare_data_for_model(
        X_train, X_test, y_train, y_test, max_seq_length, num_classes
    )
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_seq)
    X_test_tensor = torch.FloatTensor(X_test_seq)
    y_train_tensor = torch.LongTensor(y_train_seq)  # Use the windowed labels
    y_test_tensor = torch.LongTensor(y_test_seq)    # Use the windowed labels
    
    print(f"\nPyTorch tensor shapes:")
    print(f"  X_train: {X_train_tensor.shape}")
    print(f"  X_test: {X_test_tensor.shape}")
    print(f"  y_train: {y_train_tensor.shape}")
    print(f"  y_test: {y_test_tensor.shape}")
    print(f"  Device: {device}")
    
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, scaler


def create_data_loaders(X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, batch_size=32):
    """
    Create PyTorch DataLoaders for training and testing.
    """
    from torch.utils.data import TensorDataset, DataLoader
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\nDataLoaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    print(f"  Batch size: {batch_size}")
    
    return train_loader, test_loader


if __name__ == "__main__":
    # Test the data loader
    print("Testing data loader...\n")
    X_train, X_test, y_train, y_test, class_to_idx = load_datasets()
    
    print("\n" + "="*50)
    print("Preparing data for model...")
    X_train_seq, X_test_seq, y_train_seq, y_test_seq, scaler = prepare_data_for_model(
        X_train, X_test, y_train, y_test, max_seq_length=100, num_classes=6
    )
    
    print("\nData loading successful!")
