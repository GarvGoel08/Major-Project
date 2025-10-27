import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# GPU device detection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# A custom layer for Transformer-style self-attention
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        # Feed forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        
        # Layer normalization
        self.layernorm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention with residual connection
        attn_output, _ = self.attention(x, x, x)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)
        
        # Feed forward with residual connection
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

# The main Hybrid Ensemble Model
class HybridEnsembleModel(nn.Module):
    def __init__(self, input_shape, num_classes, max_seq_length, embed_dim, num_heads, ff_dim):
        super(HybridEnsembleModel, self).__init__()
        self.max_seq_length = max_seq_length
        self.num_features = input_shape[-1]  # Last dimension is features
        self.embed_dim = embed_dim
        
        # 1. CNN Branch
        self.cnn_conv1 = nn.Conv1d(self.num_features, 64, kernel_size=3, padding=1)
        self.cnn_pool1 = nn.MaxPool1d(kernel_size=2)
        self.cnn_conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.cnn_pool2 = nn.MaxPool1d(kernel_size=2)
        
        # Calculate CNN output size after pooling
        cnn_output_length = max_seq_length // 4  # Two pooling layers with stride 2
        self.cnn_flatten_size = 128 * cnn_output_length
        
        # 2. LSTM Branch
        self.lstm1 = nn.LSTM(self.num_features, 64, batch_first=True)
        self.lstm2 = nn.LSTM(64, 32, batch_first=True)
        
        # 3. Transformer Branch
        self.transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
        
        # Fusion layer
        fusion_input_size = self.cnn_flatten_size + 32 + embed_dim  # CNN + LSTM + Transformer
        self.fusion = nn.Linear(fusion_input_size, 128)
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        batch_size = x.size(0)
        
        # 1. CNN Branch
        # Input: (batch_size, seq_length, features) -> need (batch_size, features, seq_length)
        cnn_input = x.transpose(1, 2)  # (batch_size, features, seq_length)
        cnn_out = F.relu(self.cnn_conv1(cnn_input))
        cnn_out = self.cnn_pool1(cnn_out)
        cnn_out = F.relu(self.cnn_conv2(cnn_out))
        cnn_out = self.cnn_pool2(cnn_out)
        cnn_out = cnn_out.view(batch_size, -1)  # Flatten
        
        # 2. LSTM Branch
        lstm_out, (h1, c1) = self.lstm1(x)
        lstm_out, (h2, c2) = self.lstm2(lstm_out)
        lstm_out = h2[-1]  # Take the last hidden state
        
        # 3. Transformer Branch
        transformer_out = self.transformer_block(x)
        transformer_out = torch.mean(transformer_out, dim=1)  # Global average pooling
        
        # Ensemble Fusion
        concatenated = torch.cat([cnn_out, lstm_out, transformer_out], dim=1)
        fused = self.fusion(concatenated)
        
        # Final classification
        output = self.classifier(fused)
        return output

def build_hybrid_model(input_shape, num_classes, max_seq_length, embed_dim, num_heads, ff_dim):
    """Create and return the PyTorch model"""
    model = HybridEnsembleModel(input_shape, num_classes, max_seq_length, embed_dim, num_heads, ff_dim)
    model = model.to(device)  # Move model to GPU if available
    return model

# Define model parameters
max_seq_length = 500  # Example, adjust based on your signal length
num_features = 18  # Number of input features: Voltage1-3, Current1-3, ActivePower1-3, ReactivePower1-3, VoltageTHD1-3, CurrentTHD1-3
num_classes = 6  # Output classes: Normal, Overload, LLG, LL, LLLG, LG
embed_dim = num_features  # For the Transformer, should be equal to num_features
num_heads = 2
ff_dim = 32

if __name__ == "__main__":
    # Instantiate the model
    model = build_hybrid_model(
        input_shape=(max_seq_length, num_features),
        num_classes=num_classes,
        max_seq_length=max_seq_length,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim
    )
    
    # Print model summary
    print(f"\nModel created on device: {device}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test with dummy input
    dummy_input = torch.randn(1, max_seq_length, num_features).to(device)
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Model output shape: {output.shape}")
        print(f"Output probabilities: {F.softmax(output, dim=1)}")