# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Self-Attention Block (for Transformer) ---
class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        # PyTorch MultiheadAttention
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, embed_dim)
        # Note: If attn_mask is not passed, it defaults to False
        attn_output, attn_weights = self.attn(x, x, x, need_weights=False) 
        # Residual connection and normalization
        x = x + self.dropout(attn_output)
        return self.norm(x)

# --- Transformer Layer ---
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = SelfAttentionBlock(embed_dim, num_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Attention and residual
        x = self.attention(x)
        
        # Feed-forward and residual
        res = x
        x = self.ffn(x)
        x = res + self.dropout(x)
        return self.norm(x)

# --- Hybrid Fault Predictor (Ensemble Model) ---
class HybridFaultPredictor(nn.Module):
    # CHANGED transformer_heads from 4 to 6
    def __init__(self, input_size, num_classes, 
                 transformer_heads=6, transformer_ff_dim=32, 
                 cnn_filters=64, lstm_units=32, dropout=0.2):
        super().__init__()
        
        # Check for attention divisibility requirement
        if input_size % transformer_heads != 0:
            print(f"ERROR: input_size ({input_size}) must be divisible by transformer_heads ({transformer_heads}). Please adjust.")
        
        # 1. Transformer Branch
        self.transformer_block = TransformerBlock(input_size, transformer_heads, transformer_ff_dim, dropout)
        self.transformer_flatten = nn.Flatten()

        # 2. CNN Branch (Captures local feature patterns)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=cnn_filters, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=cnn_filters, out_channels=cnn_filters // 2, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1) # Global Max Pooling
        )
        
        # 3. LSTM Branch (Temporal dependencies - even for T=1, acts as feature transformation)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=lstm_units, 
                             batch_first=True, bidirectional=False)
        
        # 4. Ensemble Fusion and Classification Head
        # Total features after fusion: (input_size) + (cnn_filters/2) + (lstm_units)
        fusion_size = input_size + (cnn_filters // 2) + lstm_units
        self.classifier = nn.Sequential(
            nn.Linear(fusion_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # Input x shape: (batch_size, time_steps=1, n_features)

        # 1. Transformer Branch
        transformer_out = self.transformer_block(x)
        transformer_out = self.transformer_flatten(transformer_out) 

        # 2. CNN Branch
        cnn_input = x.transpose(1, 2) 
        cnn_out = self.cnn(cnn_input)
        cnn_out = cnn_out.squeeze(-1) # shape: (batch_size, cnn_filters/2)
        
        # 3. LSTM Branch
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :] # shape: (batch_size, lstm_units)
        
        # 4. Ensemble Fusion
        fused_features = torch.cat([transformer_out, cnn_out, lstm_out], dim=1)
        
        # 5. Classification
        output = self.classifier(fused_features)
        return output