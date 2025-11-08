import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F

# GPU device detection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Reproducible seed for deterministic splits and shuffling
SEED = 1234
import random
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

def load_datasets(data_dir='Datasets/Train'):
    """
    Load datasets from the specified directory.

    Behavior:
    - If CSV files are present, they are assumed to contain features in the first 18 columns
      and the fault/type label in the last column. All CSVs will be concatenated and the
      label values will be converted to integer classes dynamically.
    - If no CSVs are found, the function falls back to the previous Excel per-file
      mapping behavior (for backward compatibility).

    Returns:
        X_train, X_test, y_train, y_test, class_to_idx
    """

    # Expected column names (18 features)
    expected_columns = [
        'Voltage1', 'Voltage2', 'Voltage3',
        'Current1', 'Current2', 'Current3',
        'ActivePower1', 'ActivePower2', 'ActivePower3',
        'ReactivePower1', 'ReactivePower2', 'ReactivePower3',
        'VoltageTHD1', 'VoltageTHD2', 'VoltageTHD3',
        'CurrentTHD1', 'CurrentTHD2', 'CurrentTHD3'
    ]

    # Discover files in data_dir
    files = os.listdir(data_dir) if os.path.exists(data_dir) else []
    csv_files = [f for f in files if f.lower().endswith('.csv')]

    if len(csv_files) > 0:
        # Load and concatenate CSV files. Assume label is in the last column and
        # the first 18 columns are features.
        print(f"Found CSV(s): {csv_files}. Loading as combined dataset...")
        data_list = []
        label_list = []

        for fname in csv_files:
            fpath = os.path.join(data_dir, fname)
            print(f"  Reading {fname}...")
            df = pd.read_csv(fpath)
            # Limit to first 20000 rows to avoid OOM in case of huge files
            df = df.head(20000)

            if df.shape[1] < 2:
                print(f"  Warning: {fname} has <2 columns, skipping.")
                continue

            # Use first 18 columns as features if available, otherwise use all but last
            if df.shape[1] >= 19:
                df_features = df.iloc[:, :18].copy()
                labels = df.iloc[:, -1].values
            else:
                # fewer than 19 columns: assume last column is label and the rest are features
                df_features = df.iloc[:, :-1].copy()
                labels = df.iloc[:, -1].values

            # Keep a copy of the original feature cells so we can report exact values
            df_features_raw = df_features.copy()

            # Parse and convert any string/complex-looking entries into floats.
            # Examples: '125.767313388464+0i' => 125.767313388464
            def _parse_cell_to_float(v):
                # pass through numeric types
                if v is None:
                    return np.nan
                if isinstance(v, (float, int, np.floating, np.integer)):
                    try:
                        return float(v)
                    except Exception:
                        return np.nan
                s = str(v).strip()
                if s == '':
                    return np.nan
                # try direct float
                try:
                    return float(s)
                except Exception:
                    pass
                # handle complex with i (replace i->j for Python complex)
                if 'i' in s:
                    try:
                        s2 = s.replace('i', 'j')
                        c = complex(s2)
                        return float(c.real)
                    except Exception:
                        pass
                # fallback: extract first numeric token
                try:
                    import re
                    m = re.search(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", s)
                    if m:
                        return float(m.group(0))
                except Exception:
                    pass
                return np.nan

            # Apply parsing to the dataframe of features (use map to avoid FutureWarning)
            df_features = df_features.apply(lambda col: col.map(_parse_cell_to_float))
            features = df_features.values.astype(float)

            # If any NaNs introduced, fill with column mean (or 0 if the column is all NaN)
            if np.isnan(features).any():
                col_means = np.nanmean(features, axis=0)
                # replace NaN means with 0.0 for columns where mean is NaN
                col_means = np.where(np.isnan(col_means), 0.0, col_means)

                inds = np.where(np.isnan(features))

                # Print detailed info for the first several NaN occurrences
                max_reports = 50
                reported = 0
                n_rows, n_cols = features.shape
                for r, c in zip(*inds):
                    if reported >= max_reports:
                        break
                    reported += 1
                    orig_val = df_features_raw.iat[r, c] if (r < df_features_raw.shape[0] and c < df_features_raw.shape[1]) else None
                    # label for this row (from labels array for this file)
                    lbl = labels[r] if (r < len(labels)) else None
                    col_name = df_features.columns[c] if c < len(df_features.columns) else f"col_{c}"
                    replacement = float(col_means[c])
                    replaced_with_zero = replacement == 0.0
                    print(f"  NaN found in file '{fname}' - row_index={r}, original_df_index={df_features_raw.index[r]}, column='{col_name}', original_value={repr(orig_val)}, label={repr(lbl)}, replacement={replacement}, replaced_with_zero={replaced_with_zero}")

                # perform replacement
                features[inds] = np.take(col_means, inds[1])

            data_list.append(features)
            label_list.append(labels)

        if len(data_list) == 0:
            raise RuntimeError(f"No usable CSV data found in {data_dir}")

        X_all = np.vstack(data_list)
        y_all_raw = np.hstack(label_list)

        # Normalize label values to strings and create mapping
        y_all_raw = np.array([str(x).strip() for x in y_all_raw])
        unique_labels = np.unique(y_all_raw)
        class_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        y_all = np.array([class_to_idx[lbl] for lbl in y_all_raw])

        # Per-class contiguous split: for each class, take the first (1-test_frac) portion as train
        # and the remainder as test. This avoids scattering near-duplicate windows across splits.
        n = len(X_all)
        test_frac = 0.3
        train_indices = []
        test_indices = []

        # use y_all_raw (string labels) to find original positions per class
        for label in unique_labels:
            idxs = np.where(y_all_raw == label)[0]
            if len(idxs) == 0:
                continue
            n_cls = len(idxs)
            cut = int((1.0 - test_frac) * n_cls)
            # ensure at least one sample in train and test when possible
            if cut < 1 and n_cls > 1:
                cut = 1
            train_indices.extend(idxs[:cut].tolist())
            test_indices.extend(idxs[cut:].tolist())

        # sort indices to preserve ordering within each split
        train_indices = np.array(sorted(train_indices))
        test_indices = np.array(sorted(test_indices))

        X_train = X_all[train_indices]
        X_test = X_all[test_indices]
        y_train = y_all[train_indices]
        y_test = y_all[test_indices]

        print(f"  Total rows: {n}. Train: {len(X_train)}, Test: {len(X_test)}")
        print("  Classes:")
        for label, idx in class_to_idx.items():
            print(f"    {label}: {np.sum(y_train == idx)} (train), total={np.sum(y_all == idx)}")

        return X_train, X_test, y_train, y_test, class_to_idx

    # --- Fallback: per-file Excel mapping (legacy behavior) ---
    print("No CSV files found; falling back to Excel per-file mapping (legacy).")

    file_label_mapping = {
        'NormalParameters.xlsx': 'Normal',
        'Overload.xlsx': 'Overload',
        'LLG.xlsx': 'LLG',
        'LL.xlsx': 'LL',
        'LLLG.xlsx': 'LLLG',
        'LG.xlsx': 'LG'
    }

    class_to_idx = {
        'Normal': 0,
        'Overload': 1,
        'LLG': 2,
        'LL': 3,
        'LLLG': 4,
        'LG': 5
    }

    all_data = []

    for filename, label in file_label_mapping.items():
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            print(f"Loading {filename}...")
            df = pd.read_excel(filepath)
            df = df.head(2000)

            # If column names don't match, assume the useful columns start at index 2
            if list(df.columns) != expected_columns and df.shape[1] >= 20:
                print(f"  Warning: Column names don't match expected. Using columns 2..19 as features.")
                df = df.iloc[:, 2:20]
                df.columns = expected_columns

            data = df[expected_columns].values
            n_samples = len(data)
            train_size = min(1000, n_samples)
            train_data = data[:train_size]
            test_data = data[train_size:]

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

    if len(all_data) == 0:
        raise RuntimeError(f"No datasets found in {data_dir}")

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
    def create_windows(X, y, window_size, for_train=True):
        if len(X) < window_size:
            # If we have fewer samples than window size, pad with last values
            padding_needed = window_size - len(X)
            X_padded = np.vstack([X, np.tile(X[-1], (padding_needed, 1))])
            return X_padded[np.newaxis, :, :], np.array([y[0]])
        else:
            # Create overlapping windows
            windows = []
            labels = []
            # Choose step size: more overlap for training, non-overlapping for test to avoid leakage
            if for_train:
                step_size = max(1, window_size // 8)  # ~87.5% overlap (augment)
            else:
                step_size = max(1, window_size)  # non-overlapping test windows
            
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
            
            X_win, y_win = create_windows(class_X_train, class_y_train, max_seq_length, for_train=True)
            X_train_windows.append(X_win)
            y_train_windows.append(y_win)
        
        # Test data for this class
        test_mask = y_test == class_idx
        if np.any(test_mask):
            class_X_test = X_test_scaled[test_mask]
            class_y_test = y_test[test_mask]
            
            X_win, y_win = create_windows(class_X_test, class_y_test, max_seq_length, for_train=False)
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
        nan_indices = np.where(nan_train_windows)[0]
        print(f"  Warning: {len(nan_indices)} training windows contain NaNs. Indices: {nan_indices[:10]}")
        # Print detailed info for first few windows containing NaNs
        for win_idx in nan_indices[:10]:
            nan_positions = np.argwhere(np.isnan(X_train_seq[win_idx]))
            label_for_window = y_train_windows[0][0] if False else (y_train_seq[win_idx] if 'y_train_seq' in locals() else None)
            print(f"    Window {win_idx}: {nan_positions.shape[0]} NaN positions, label={label_for_window}")
            # show up to first 10 positions
            for pos in nan_positions[:10]:
                t, f = int(pos[0]), int(pos[1])
                print(f"      timestep={t}, feature={f}")
        # Replace NaNs with zero (safe fallback) and continue
        X_train_seq[np.isnan(X_train_seq)] = 0.0

    nan_test_windows = np.isnan(X_test_seq).any(axis=(1, 2))
    if np.any(nan_test_windows):
        nan_indices = np.where(nan_test_windows)[0]
        print(f"  Warning: {len(nan_indices)} test windows contain NaNs. Indices: {nan_indices[:10]}")
        for win_idx in nan_indices[:10]:
            nan_positions = np.argwhere(np.isnan(X_test_seq[win_idx]))
            label_for_window = y_test_seq[win_idx] if 'y_test_seq' in locals() else None
            print(f"    Window {win_idx}: {nan_positions.shape[0]} NaN positions, label={label_for_window}")
            for pos in nan_positions[:10]:
                t, f = int(pos[0]), int(pos[1])
                print(f"      timestep={t}, feature={f}")
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
