# Module 5: Deep Learning for Financial Time Series

## Table of Contents
1. [LSTM and GRU Networks](#lstm-and-gru-networks)
2. [Transformer Models for Finance](#transformer-models-for-finance)
3. [CNN-based Time Series Models](#cnn-based-time-series-models)
4. [Hybrid Architectures](#hybrid-architectures)
5. [Advanced Techniques](#advanced-techniques)
6. [PhD-Level Research Topics](#phd-level-research-topics)

## LSTM and GRU Networks

### Introduction to Recurrent Architectures

Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) networks are specialized recurrent neural networks designed to capture temporal dependencies in sequential data. In financial time series, they excel at modeling price momentum, volatility clustering, and regime changes.

### LSTM Architecture for Price Prediction

```python
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Tuple, Optional

class FinancialLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout: float = 0.2,
        bidirectional: bool = False
    ):
        super(FinancialLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_size,
            num_heads=4,
            dropout=dropout
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        
        h0 = torch.zeros(
            self.num_layers * (2 if self.bidirectional else 1),
            batch_size,
            self.hidden_size
        ).to(x.device)
        
        c0 = torch.zeros(
            self.num_layers * (2 if self.bidirectional else 1),
            batch_size,
            self.hidden_size
        ).to(x.device)
        
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        final_hidden = attn_out[:, -1, :]
        
        output = self.fc_layers(final_hidden)
        
        return output
```

### Multi-Step Ahead Forecasting

```python
class MultiHorizonLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        forecast_horizons: list,
        dropout: float = 0.2
    ):
        super(MultiHorizonLSTM, self).__init__()
        
        self.forecast_horizons = forecast_horizons
        
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.horizon_heads = nn.ModuleDict({
            f'horizon_{h}': nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, 1)
            ) for h in forecast_horizons
        })
        
    def forward(self, x: torch.Tensor) -> dict:
        _, (hn, _) = self.encoder(x)
        
        final_hidden = hn[-1]
        
        predictions = {}
        for h in self.forecast_horizons:
            predictions[f'horizon_{h}'] = self.horizon_heads[f'horizon_{h}'](final_hidden)
        
        return predictions
```

### Handling Long-Term Dependencies

```python
class ResidualLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout: float = 0.2
    ):
        super(ResidualLSTM, self).__init__()
        
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        self.output_layer = nn.Linear(hidden_size, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        
        for lstm, ln in zip(self.lstm_layers, self.layer_norms):
            residual = x
            x, _ = lstm(x)
            x = ln(x + residual)
            x = self.dropout(x)
        
        output = self.output_layer(x[:, -1, :])
        
        return output
```

### GRU Architecture

```python
class FinancialGRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout: float = 0.2
    ):
        super(FinancialGRU, self).__init__()
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gru_out, _ = self.gru(x)
        
        attention_weights = torch.softmax(
            self.attention(gru_out).squeeze(-1), dim=1
        )
        
        context = torch.sum(gru_out * attention_weights.unsqueeze(-1), dim=1)
        
        output = self.fc(context)
        
        return output
```

## Transformer Models for Finance

### Self-Attention Mechanism

The attention mechanism allows the model to focus on relevant time steps regardless of their position in the sequence. For financial data, this is crucial for capturing sudden market shifts and long-range dependencies.

**Attention Formula:**
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### Vanilla Transformer for Time Series

```python
class FinancialTransformer(nn.Module):
    def __init__(
        self,
        input_size: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        dim_feedforward: int,
        output_size: int,
        dropout: float = 0.1,
        max_seq_length: int = 5000
    ):
        super(FinancialTransformer, self).__init__()
        
        self.d_model = d_model
        
        self.input_projection = nn.Linear(input_size, d_model)
        
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_seq_length)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, output_size)
        )
        
    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        src = self.input_projection(src) * np.sqrt(self.d_model)
        
        src = self.positional_encoding(src)
        
        transformer_out = self.transformer_encoder(src, src_mask)
        
        output = self.output_layer(transformer_out[:, -1, :])
        
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
```

### Temporal Fusion Transformer (TFT)

```python
class TemporalFusionTransformer(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_heads: int,
        num_encoder_layers: int,
        dropout: float = 0.1
    ):
        super(TemporalFusionTransformer, self).__init__()
        
        self.variable_selection = VariableSelectionNetwork(
            input_size, hidden_size, dropout
        )
        
        self.lstm_encoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        
        self.gating_layer = GatedLinearUnit(hidden_size, dropout)
        
        self.static_enrichment = GatedResidualNetwork(
            hidden_size, hidden_size, dropout
        )
        
        self.temporal_self_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.position_wise_ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_selected = self.variable_selection(x)
        
        lstm_out, _ = self.lstm_encoder(x_selected)
        
        lstm_out = self.gating_layer(lstm_out)
        
        enriched = self.static_enrichment(lstm_out)
        
        attn_out, attn_weights = self.temporal_self_attention(
            enriched, enriched, enriched
        )
        
        attn_out = self.layer_norm(attn_out + enriched)
        
        ff_out = self.position_wise_ff(attn_out)
        ff_out = self.layer_norm(ff_out + attn_out)
        
        output = self.output_layer(ff_out[:, -1, :])
        
        return output


class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.1):
        super(VariableSelectionNetwork, self).__init__()
        
        self.grn = GatedResidualNetwork(input_size, hidden_size, dropout)
        
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.grn(x.mean(dim=1, keepdim=True).expand_as(x))
        weights = self.softmax(weights)
        
        return x * weights


class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.1):
        super(GatedResidualNetwork, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.gate = GatedLinearUnit(hidden_size, dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        if input_size != hidden_size:
            self.skip_connection = nn.Linear(input_size, hidden_size)
        else:
            self.skip_connection = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.skip_connection is None else self.skip_connection(x)
        
        x = self.fc1(x)
        x = self.elu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        
        x = self.gate(x)
        
        x = self.layer_norm(x + residual)
        
        return x


class GatedLinearUnit(nn.Module):
    def __init__(self, input_size: int, dropout: float = 0.1):
        super(GatedLinearUnit, self).__init__()
        
        self.fc = nn.Linear(input_size, input_size * 2)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = self.dropout(x)
        
        values, gates = torch.chunk(x, 2, dim=-1)
        
        return values * self.sigmoid(gates)
```

### Interpretable Attention Weights

```python
def visualize_attention_weights(
    model: nn.Module,
    input_data: torch.Tensor,
    feature_names: list,
    time_steps: list
):
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    model.eval()
    with torch.no_grad():
        output, attention_weights = model(input_data, return_attention=True)
    
    attention_weights = attention_weights.squeeze().cpu().numpy()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        attention_weights,
        xticklabels=time_steps,
        yticklabels=feature_names,
        cmap='YlOrRd',
        cbar_kws={'label': 'Attention Weight'}
    )
    plt.title('Temporal Attention Weights')
    plt.xlabel('Time Steps')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.show()
```

## CNN-based Time Series Models

### Temporal Convolutional Networks (TCN)

```python
class TemporalConvNet(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_channels: list,
        kernel_size: int = 2,
        dropout: float = 0.2
    ):
        super(TemporalConvNet, self).__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size-1) * dilation_size,
                    dropout=dropout
                )
            )
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TemporalBlock(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2
    ):
        super(TemporalBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous()
```

### WaveNet Architecture

```python
class WaveNet(nn.Module):
    def __init__(
        self,
        input_channels: int,
        residual_channels: int,
        skip_channels: int,
        output_channels: int,
        num_blocks: int = 4,
        num_layers_per_block: int = 10,
        kernel_size: int = 2
    ):
        super(WaveNet, self).__init__()
        
        self.input_conv = nn.Conv1d(input_channels, residual_channels, 1)
        
        self.residual_blocks = nn.ModuleList()
        
        for b in range(num_blocks):
            for i in range(num_layers_per_block):
                dilation = 2 ** i
                self.residual_blocks.append(
                    ResidualBlock(
                        residual_channels,
                        skip_channels,
                        kernel_size,
                        dilation
                    )
                )
        
        self.output_conv1 = nn.Conv1d(skip_channels, skip_channels, 1)
        self.output_conv2 = nn.Conv1d(skip_channels, output_channels, 1)
        
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_conv(x)
        
        skip_connections = []
        for block in self.residual_blocks:
            x, skip = block(x)
            skip_connections.append(skip)
        
        x = torch.sum(torch.stack(skip_connections), dim=0)
        
        x = self.relu(x)
        x = self.output_conv1(x)
        x = self.relu(x)
        x = self.output_conv2(x)
        
        return x


class ResidualBlock(nn.Module):
    def __init__(
        self,
        residual_channels: int,
        skip_channels: int,
        kernel_size: int,
        dilation: int
    ):
        super(ResidualBlock, self).__init__()
        
        self.dilated_conv = nn.Conv1d(
            residual_channels,
            residual_channels * 2,
            kernel_size,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation
        )
        
        self.residual_conv = nn.Conv1d(residual_channels, residual_channels, 1)
        self.skip_conv = nn.Conv1d(residual_channels, skip_channels, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        conv_out = self.dilated_conv(x)
        conv_out = conv_out[:, :, :x.size(2)]
        
        gate, filter_out = torch.chunk(conv_out, 2, dim=1)
        
        gated = torch.tanh(filter_out) * torch.sigmoid(gate)
        
        residual = self.residual_conv(gated)
        skip = self.skip_conv(gated)
        
        return (x + residual), skip
```

## Hybrid Architectures

### CNN-LSTM Combination

```python
class CNNLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_filters: int,
        kernel_size: int,
        lstm_hidden_size: int,
        num_lstm_layers: int,
        output_size: int,
        dropout: float = 0.2
    ):
        super(CNNLSTM, self).__init__()
        
        self.conv1 = nn.Conv1d(input_size, num_filters, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(num_filters, num_filters*2, kernel_size, padding=kernel_size//2)
        self.pool = nn.MaxPool1d(2)
        self.batch_norm1 = nn.BatchNorm1d(num_filters)
        self.batch_norm2 = nn.BatchNorm1d(num_filters*2)
        
        self.lstm = nn.LSTM(
            input_size=num_filters*2,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_size, lstm_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_size // 2, output_size)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        
        x = torch.relu(self.batch_norm1(self.conv1(x)))
        x = self.pool(x)
        x = torch.relu(self.batch_norm2(self.conv2(x)))
        x = self.pool(x)
        
        x = x.transpose(1, 2)
        
        lstm_out, _ = self.lstm(x)
        
        output = self.fc(lstm_out[:, -1, :])
        
        return output
```

### Attention-Augmented RNN

```python
class AttentionLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout: float = 0.2
    ):
        super(AttentionLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.attention = BahdanauAttention(hidden_size)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        lstm_out, _ = self.lstm(x)
        
        context, attention_weights = self.attention(lstm_out)
        
        output = self.fc(context)
        
        return output, attention_weights


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super(BahdanauAttention, self).__init__()
        
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)
        
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        score = self.Va(torch.tanh(self.Wa(hidden_states)))
        
        attention_weights = torch.softmax(score, dim=1)
        
        context = torch.sum(hidden_states * attention_weights, dim=1)
        
        return context, attention_weights
```

## Advanced Techniques

### Transfer Learning in Finance

```python
class FinancialTransferLearning:
    def __init__(self, pretrained_model: nn.Module):
        self.pretrained_model = pretrained_model
        
    def freeze_layers(self, num_layers_to_freeze: int):
        layers = list(self.pretrained_model.children())
        
        for layer in layers[:num_layers_to_freeze]:
            for param in layer.parameters():
                param.requires_grad = False
                
    def fine_tune(
        self,
        train_loader,
        val_loader,
        num_epochs: int = 10,
        learning_rate: float = 1e-4
    ):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pretrained_model.to(device)
        
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.pretrained_model.parameters()),
            lr=learning_rate
        )
        
        criterion = nn.MSELoss()
        
        for epoch in range(num_epochs):
            self.pretrained_model.train()
            train_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = self.pretrained_model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            val_loss = self._validate(val_loader, criterion, device)
            
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}')
            
    def _validate(self, val_loader, criterion, device):
        self.pretrained_model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = self.pretrained_model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        return val_loss / len(val_loader)
```

### Meta-Learning for Few-Shot Prediction

```python
class MAML(nn.Module):
    def __init__(self, model: nn.Module, inner_lr: float = 0.01, outer_lr: float = 0.001):
        super(MAML, self).__init__()
        
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=outer_lr)
        
    def inner_loop(self, support_x, support_y, num_steps: int = 5):
        fast_weights = {name: param.clone() for name, param in self.model.named_parameters()}
        
        for _ in range(num_steps):
            pred = self.model(support_x)
            loss = nn.functional.mse_loss(pred, support_y)
            
            grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)
            
            fast_weights = {
                name: param - self.inner_lr * grad
                for (name, param), grad in zip(fast_weights.items(), grads)
            }
        
        return fast_weights
    
    def outer_loop(self, query_x, query_y, fast_weights):
        pred = self.model(query_x)
        loss = nn.functional.mse_loss(pred, query_y)
        
        self.meta_optimizer.zero_grad()
        loss.backward()
        self.meta_optimizer.step()
        
        return loss.item()
```

### Uncertainty Quantification

```python
class BayesianLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout: float = 0.2
    ):
        super(BayesianLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        
        self.mu_layer = nn.Linear(hidden_size, output_size)
        self.sigma_layer = nn.Linear(hidden_size, output_size)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        lstm_out, _ = self.lstm(x)
        
        hidden = self.dropout(lstm_out[:, -1, :])
        
        mu = self.mu_layer(hidden)
        log_sigma = self.sigma_layer(hidden)
        sigma = torch.exp(log_sigma)
        
        return mu, sigma
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        num_samples: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.train()
        
        predictions = []
        for _ in range(num_samples):
            mu, sigma = self.forward(x)
            sample = torch.normal(mu, sigma)
            predictions.append(sample)
        
        predictions = torch.stack(predictions)
        
        mean_prediction = predictions.mean(dim=0)
        std_prediction = predictions.std(dim=0)
        
        return mean_prediction, std_prediction
```

## PhD-Level Research Topics

### State Space Models for Finance (Mamba, S4)

State Space Models (SSMs) represent a paradigm shift in sequence modeling, offering linear scaling with sequence length while maintaining the expressiveness of attention mechanisms. This is crucial for high-frequency financial data.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math

class S4DKernel(nn.Module):
    """Structured State Space for Sequences (S4D) - Diagonal variant"""
    def __init__(self, d_model: int, N: int = 64, dt_min: float = 0.001, dt_max: float = 0.1):
        super().__init__()
        self.N = N
        self.d_model = d_model
        
        # Diagonal state matrix (HiPPO initialization)
        A = repeat(torch.arange(1, N + 1), 'n -> h n', h=d_model).clone().contiguous()
        A = -0.5 + 1j * math.pi * A
        self.register_buffer('A', A)
        
        # Input matrix
        self.B = nn.Parameter(torch.randn(d_model, N) * 0.02)
        
        # Output matrix
        self.C = nn.Parameter(torch.randn(d_model, N) * 0.02)
        
        # Learnable timestep
        log_dt = torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)
        
    def forward(self, L: int):
        dt = torch.exp(self.log_dt)
        
        # Discretize A using ZOH
        dtA = dt.unsqueeze(-1) * self.A
        K = dtA.unsqueeze(-1) * torch.arange(L, device=self.A.device)
        C_expanded = self.C.unsqueeze(-1)
        B_expanded = self.B.unsqueeze(-1)
        
        # Compute kernel
        kernel = torch.sum(C_expanded * torch.exp(K) * B_expanded, dim=1).real
        return kernel


class MambaBlock(nn.Module):
    """Mamba Block - Selective State Space Model for Financial Time Series"""
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        
        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Convolution for local context
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, 
            kernel_size=d_conv, padding=d_conv - 1, groups=self.d_inner
        )
        
        # SSM parameters (selective mechanism)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(d_state, self.d_inner, bias=True)
        
        # State matrices
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32), 'n -> d n', d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        
        # Input projection and split
        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1)
        
        # Convolution
        x_conv = rearrange(x_proj, 'b l d -> b d l')
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        x_conv = rearrange(x_conv, 'b d l -> b l d')
        x_conv = F.silu(x_conv)
        
        # Selective SSM
        x_ssm = self.x_proj(x_conv)
        dt, B, C = torch.split(x_ssm, [1, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(F.softplus(dt.squeeze(-1))))
        
        # Discretize
        A = -torch.exp(self.A_log)
        dA = torch.exp(dt.unsqueeze(-1) * A)
        dB = dt.unsqueeze(-1) * B.unsqueeze(2)
        
        # Selective scan (simplified)
        y = self._selective_scan(x_conv, dA, dB, C)
        
        # Skip connection and output
        y = y + self.D * x_conv
        y = y * F.silu(z)
        
        return self.out_proj(y)
    
    def _selective_scan(self, x, dA, dB, C):
        batch, seq_len, d_inner = x.shape
        d_state = dA.shape[-1]
        
        # Initialize state
        h = torch.zeros(batch, d_inner, d_state, device=x.device)
        
        outputs = []
        for t in range(seq_len):
            h = dA[:, t] * h + dB[:, t] * x[:, t].unsqueeze(-1)
            y = torch.sum(h * C[:, t].unsqueeze(1), dim=-1)
            outputs.append(y)
        
        return torch.stack(outputs, dim=1)


class FinancialMamba(nn.Module):
    """Mamba-based model for financial time series prediction"""
    def __init__(
        self, input_size: int, d_model: int, n_layers: int, 
        output_size: int, d_state: int = 16
    ):
        super().__init__()
        
        self.embedding = nn.Linear(input_size, d_model)
        
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                MambaBlock(d_model, d_state)
            )
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        
        for layer in self.layers:
            x = x + layer(x)
        
        x = self.norm(x[:, -1])
        return self.output(x)
```

### Foundation Models for Financial Time Series

Foundation models pretrained on massive time series datasets represent the frontier of financial forecasting, enabling zero-shot and few-shot prediction.

```python
class TimeSeriesFoundationModel(nn.Module):
    """Foundation model architecture for time series (similar to TimeGPT, Lag-Llama)"""
    def __init__(
        self, 
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 12,
        max_seq_len: int = 2048,
        patch_size: int = 16,
        n_freq_features: int = 64
    ):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        
        # Patch embedding (treating time series as patches similar to ViT)
        self.patch_embed = nn.Linear(patch_size, d_model)
        
        # Frequency features (Fourier basis)
        self.freq_embed = nn.Linear(n_freq_features, d_model)
        
        # Learnable position embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len // patch_size, d_model) * 0.02)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=0.1, activation='gelu', batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        
        # Output head for probabilistic prediction
        self.mean_head = nn.Linear(d_model, patch_size)
        self.std_head = nn.Linear(d_model, patch_size)
        
    def _extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        n_patches = L // self.patch_size
        x = x[:, :n_patches * self.patch_size].reshape(B, n_patches, self.patch_size * C)
        return x
    
    def _compute_freq_features(self, x: torch.Tensor) -> torch.Tensor:
        fft = torch.fft.rfft(x, dim=1)
        freq_features = torch.cat([fft.real, fft.imag], dim=-1)
        return freq_features[:, :64]
    
    def forward(
        self, x: torch.Tensor, 
        return_distribution: bool = True
    ) -> tuple:
        B = x.shape[0]
        
        # Extract patches
        patches = self._extract_patches(x)
        patch_emb = self.patch_embed(patches)
        
        # Add positional embeddings
        patch_emb = patch_emb + self.pos_embed[:, :patch_emb.shape[1]]
        
        # Causal mask for autoregressive generation
        seq_len = patch_emb.shape[1]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        
        # Decode
        hidden = self.decoder(patch_emb, patch_emb, tgt_mask=causal_mask)
        
        # Predict distribution parameters
        mean = self.mean_head(hidden)
        log_std = self.std_head(hidden)
        std = F.softplus(log_std) + 1e-6
        
        if return_distribution:
            return mean, std
        return mean
    
    def generate(
        self, 
        context: torch.Tensor, 
        horizon: int,
        n_samples: int = 100
    ) -> torch.Tensor:
        """Generate probabilistic forecasts"""
        self.eval()
        samples = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                current = context.clone()
                predictions = []
                
                for _ in range(horizon // self.patch_size):
                    mean, std = self.forward(current)
                    # Sample from predicted distribution
                    pred = torch.normal(mean[:, -1], std[:, -1])
                    predictions.append(pred)
                    current = torch.cat([current[:, self.patch_size:], pred.unsqueeze(1)], dim=1)
                
                samples.append(torch.cat(predictions, dim=1))
        
        return torch.stack(samples)


class LagLlamaFinance(nn.Module):
    """Lag-Llama style model with lagged features for financial forecasting"""
    def __init__(
        self, 
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 6,
        lags: list = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89],  # Fibonacci lags
        context_length: int = 512
    ):
        super().__init__()
        self.lags = lags
        self.context_length = context_length
        
        # Lag feature extractor
        self.lag_embed = nn.Linear(len(lags), d_model)
        
        # Value embedding
        self.value_embed = nn.Linear(1, d_model)
        
        # Time features
        self.time_embed = nn.Linear(4, d_model)  # hour, day, month, year
        
        # Transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads,
                dim_feedforward=d_model * 4, dropout=0.1,
                batch_first=True, norm_first=True
            ),
            num_layers=n_layers
        )
        
        # Distribution head (Student-t for heavy tails)
        self.loc_head = nn.Linear(d_model, 1)
        self.scale_head = nn.Linear(d_model, 1)
        self.df_head = nn.Linear(d_model, 1)
        
    def _compute_lag_features(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        lag_features = []
        
        for lag in self.lags:
            if lag < L:
                lagged = F.pad(x[:, :-lag], (0, 0, lag, 0))
                lag_features.append(x - lagged)
            else:
                lag_features.append(torch.zeros_like(x))
        
        return torch.cat(lag_features, dim=-1)
    
    def forward(self, x: torch.Tensor, time_features: torch.Tensor = None):
        # Compute lag features
        lag_feats = self._compute_lag_features(x)
        
        # Embeddings
        value_emb = self.value_embed(x)
        lag_emb = self.lag_embed(lag_feats)
        
        combined = value_emb + lag_emb
        
        if time_features is not None:
            time_emb = self.time_embed(time_features)
            combined = combined + time_emb
        
        # Causal mask
        seq_len = combined.shape[1]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        
        # Transform
        hidden = self.transformer(combined, mask=causal_mask)
        
        # Student-t distribution parameters
        loc = self.loc_head(hidden)
        scale = F.softplus(self.scale_head(hidden)) + 1e-6
        df = F.softplus(self.df_head(hidden)) + 2.0  # df > 2 for finite variance
        
        return loc, scale, df
```

### Neural Controlled Differential Equations for Irregular Financial Data

Neural CDEs are particularly suited for financial data with irregular sampling (tick data, market closures) and naturally handle missing data.

```python
try:
    import torchcde
    
    class NeuralCDEFinance(nn.Module):
        """Neural Controlled Differential Equations for financial time series"""
        def __init__(self, input_size: int, hidden_size: int, output_size: int):
            super().__init__()
            self.hidden_size = hidden_size
            
            # Initial hidden state network
            self.initial = nn.Linear(input_size, hidden_size)
            
            # CDE function (vector field)
            self.cde_func = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.Tanh(),
                nn.Linear(hidden_size * 2, hidden_size * input_size)
            )
            
            # Output network
            self.output = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size)
            )
            
        def forward(self, coeffs: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
            # Create cubic spline interpolation
            X = torchcde.CubicSpline(coeffs, times)
            
            # Initial hidden state
            X0 = X.evaluate(times[0])
            h0 = self.initial(X0)
            
            # Solve CDE
            def cde_func(t, h):
                Xdot = X.derivative(t)
                return torch.einsum(
                    'bh,bhd->bd',
                    self.cde_func(h).view(h.shape[0], self.hidden_size, -1),
                    Xdot
                )
            
            h_final = torchcde.cdeint(
                X=X, z0=h0, func=cde_func, t=times,
                method='rk4', options={'step_size': 0.1}
            )[-1]
            
            return self.output(h_final)
            
except ImportError:
    print("torchcde not available - Neural CDE features disabled")


class AttentionNeuralODE(nn.Module):
    """Neural ODE with attention for financial time series with regime changes"""
    def __init__(self, input_size: int, hidden_size: int, n_heads: int = 4):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Encoder for initial conditions
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        
        # ODE function with attention
        self.attention = nn.MultiheadAttention(hidden_size, n_heads, batch_first=True)
        self.ode_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.Tanh(),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        
        self.context = None
        
    def set_context(self, context: torch.Tensor):
        """Set context from encoded historical data"""
        self.context = context
        
    def ode_func(self, t: float, h: torch.Tensor) -> torch.Tensor:
        # Attend to historical context
        h_query = h.unsqueeze(1)
        attended, _ = self.attention(h_query, self.context, self.context)
        attended = attended.squeeze(1)
        
        # Compute dynamics
        combined = torch.cat([h, attended], dim=-1)
        return self.ode_net(combined)
```

### Conformal Prediction for Financial Uncertainty Quantification

Conformal prediction provides distribution-free, finite-sample valid prediction intervals - essential for risk management.

```python
class ConformalPredictor:
    """Conformal prediction for financial forecasting with guaranteed coverage"""
    def __init__(self, model: nn.Module, alpha: float = 0.1):
        self.model = model
        self.alpha = alpha
        self.calibration_scores = None
        
    def calibrate(self, cal_loader: torch.utils.data.DataLoader):
        """Calibrate on held-out calibration set"""
        self.model.eval()
        scores = []
        
        with torch.no_grad():
            for X, y in cal_loader:
                pred = self.model(X)
                # Nonconformity score (absolute residual)
                score = torch.abs(y - pred)
                scores.append(score)
        
        self.calibration_scores = torch.cat(scores)
        
        # Compute quantile for desired coverage
        n = len(self.calibration_scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.q_hat = torch.quantile(self.calibration_scores, q_level)
        
    def predict_interval(self, X: torch.Tensor) -> tuple:
        """Predict with conformal intervals"""
        self.model.eval()
        with torch.no_grad():
            pred = self.model(X)
        
        lower = pred - self.q_hat
        upper = pred + self.q_hat
        
        return pred, lower, upper
    
    
class AdaptiveConformalPredictor:
    """Adaptive conformal prediction for non-stationary financial data"""
    def __init__(self, model: nn.Module, alpha: float = 0.1, gamma: float = 0.01):
        self.model = model
        self.alpha = alpha
        self.gamma = gamma  # Learning rate for adaptive alpha
        self.alpha_t = alpha  # Adaptive alpha
        self.q_hat = None
        
    def update(self, y_true: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor):
        """Online update of coverage level"""
        # Check if true values are covered
        covered = (y_true >= lower) & (y_true <= upper)
        err_t = 1 - covered.float().mean()
        
        # Adaptive update
        self.alpha_t = self.alpha_t + self.gamma * (self.alpha - err_t)
        self.alpha_t = torch.clamp(torch.tensor(self.alpha_t), 0.01, 0.5).item()
        
    def predict_interval(self, X: torch.Tensor) -> tuple:
        self.model.eval()
        with torch.no_grad():
            mean, std = self.model(X)  # Assume model outputs distribution params
        
        # Use adaptive alpha
        z = torch.distributions.Normal(0, 1).icdf(torch.tensor(1 - self.alpha_t / 2))
        lower = mean - z * std
        upper = mean + z * std
        
        return mean, lower, upper


class QuantileRegressionEnsemble(nn.Module):
    """Ensemble of quantile regressors for prediction intervals"""
    def __init__(self, base_model_class, model_kwargs: dict, quantiles: list = [0.05, 0.5, 0.95]):
        super().__init__()
        self.quantiles = quantiles
        
        self.models = nn.ModuleList([
            base_model_class(**model_kwargs) for _ in quantiles
        ])
        
    def forward(self, x: torch.Tensor) -> dict:
        predictions = {}
        for q, model in zip(self.quantiles, self.models):
            predictions[f'q{int(q*100)}'] = model(x)
        return predictions
    
    def quantile_loss(self, y_true: torch.Tensor, predictions: dict) -> torch.Tensor:
        total_loss = 0
        for q in self.quantiles:
            pred = predictions[f'q{int(q*100)}']
            error = y_true - pred
            loss = torch.max(q * error, (q - 1) * error)
            total_loss = total_loss + loss.mean()
        return total_loss
```

### Koopman Operator Theory for Financial Dynamics

Koopman operators provide a linear representation of nonlinear dynamics, enabling spectral analysis and long-term prediction.

```python
class DeepKoopmanFinance(nn.Module):
    """Deep Koopman network for financial time series"""
    def __init__(
        self, input_dim: int, latent_dim: int, 
        encoder_layers: list = [64, 128], 
        n_modes: int = 32
    ):
        super().__init__()
        
        # Encoder: observable space -> latent Koopman space
        encoder_dims = [input_dim] + encoder_layers + [latent_dim]
        encoder_layers_list = []
        for i in range(len(encoder_dims) - 1):
            encoder_layers_list.extend([
                nn.Linear(encoder_dims[i], encoder_dims[i + 1]),
                nn.ReLU() if i < len(encoder_dims) - 2 else nn.Identity()
            ])
        self.encoder = nn.Sequential(*encoder_layers_list)
        
        # Koopman operator (learnable linear dynamics in latent space)
        self.K = nn.Linear(latent_dim, latent_dim, bias=False)
        
        # Decoder: latent space -> observable space
        decoder_dims = [latent_dim] + encoder_layers[::-1] + [input_dim]
        decoder_layers_list = []
        for i in range(len(decoder_dims) - 1):
            decoder_layers_list.extend([
                nn.Linear(decoder_dims[i], decoder_dims[i + 1]),
                nn.ReLU() if i < len(decoder_dims) - 2 else nn.Identity()
            ])
        self.decoder = nn.Sequential(*decoder_layers_list)
        
        # Auxiliary network for eigenvalue regularization
        self.eigenvalue_predictor = nn.Linear(latent_dim, n_modes * 2)
        
    def forward(self, x: torch.Tensor, n_steps: int = 1) -> tuple:
        # Encode to Koopman space
        g = self.encoder(x)
        
        # Apply Koopman operator n_steps times
        predictions = []
        g_current = g
        for _ in range(n_steps):
            g_current = self.K(g_current)
            x_pred = self.decoder(g_current)
            predictions.append(x_pred)
        
        return torch.stack(predictions, dim=1), g
    
    def compute_eigenspectrum(self) -> tuple:
        """Compute eigenvalues and eigenvectors of learned Koopman operator"""
        K_matrix = self.K.weight.data.cpu().numpy()
        eigenvalues, eigenvectors = np.linalg.eig(K_matrix)
        return eigenvalues, eigenvectors
    
    def stability_loss(self) -> torch.Tensor:
        """Regularization to encourage stable dynamics"""
        eigenvalues, _ = torch.linalg.eig(self.K.weight)
        # Penalize eigenvalues with magnitude > 1 (unstable modes)
        magnitudes = torch.abs(eigenvalues)
        return F.relu(magnitudes - 1.0).mean()
    
    def loss(self, x_seq: torch.Tensor) -> torch.Tensor:
        """Combined loss: reconstruction + prediction + stability"""
        batch, seq_len, dim = x_seq.shape
        
        total_loss = 0
        for t in range(seq_len - 1):
            x_t = x_seq[:, t]
            x_next = x_seq[:, t + 1]
            
            # One-step prediction
            pred, g = self.forward(x_t, n_steps=1)
            pred = pred.squeeze(1)
            
            # Reconstruction loss
            x_recon = self.decoder(self.encoder(x_t))
            recon_loss = F.mse_loss(x_recon, x_t)
            
            # Prediction loss
            pred_loss = F.mse_loss(pred, x_next)
            
            total_loss = total_loss + recon_loss + pred_loss
        
        # Stability regularization
        stability = self.stability_loss()
        
        return total_loss / (seq_len - 1) + 0.01 * stability
```

### Causal Inference in Time Series

```python
class CausalLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        treatment_idx: int
    ):
        super(CausalLSTM, self).__init__()
        
        self.treatment_idx = treatment_idx
        
        self.confounding_lstm = nn.LSTM(
            input_size=input_size - 1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.treatment_embedding = nn.Linear(1, hidden_size)
        
        self.outcome_lstm = nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.output_layer = nn.Linear(hidden_size, output_size)
        
    def forward(
        self,
        x: torch.Tensor,
        treatment: torch.Tensor
    ) -> torch.Tensor:
        confounding_features = torch.cat([
            x[:, :, :self.treatment_idx],
            x[:, :, self.treatment_idx+1:]
        ], dim=2)
        
        confound_out, _ = self.confounding_lstm(confounding_features)
        
        treatment_emb = self.treatment_embedding(treatment.unsqueeze(-1))
        treatment_emb = treatment_emb.unsqueeze(1).expand(-1, x.size(1), -1)
        
        combined = torch.cat([confound_out, treatment_emb], dim=2)
        
        outcome_out, _ = self.outcome_lstm(combined)
        
        output = self.output_layer(outcome_out[:, -1, :])
        
        return output
    
    def estimate_treatment_effect(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        treatment_0 = torch.zeros(x.size(0), x.size(1))
        treatment_1 = torch.ones(x.size(0), x.size(1))
        
        y0 = self.forward(x, treatment_0)
        y1 = self.forward(x, treatment_1)
        
        treatment_effect = y1 - y0
        
        return treatment_effect
```

### Neural ODEs for Time Series

```python
from torchdiffeq import odeint

class ODEFunc(nn.Module):
    def __init__(self, hidden_size: int):
        super(ODEFunc, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.Tanh(),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        
    def forward(self, t, h):
        return self.net(h)


class NeuralODE(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int
    ):
        super(NeuralODE, self).__init__()
        
        self.encoder = nn.Linear(input_size, hidden_size)
        
        self.ode_func = ODEFunc(hidden_size)
        
        self.decoder = nn.Linear(hidden_size, output_size)
        
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        h0 = self.encoder(x[:, 0, :])
        
        h_trajectory = odeint(self.ode_func, h0, t, method='dopri5')
        
        output = self.decoder(h_trajectory[-1])
        
        return output
```

## Practical Implementation

### Complete Training Pipeline

```python
class DeepLearningPipeline:
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5
    ):
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=5,
            factor=0.5
        )
        
        self.criterion = nn.MSELoss()
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.early_stopping_patience = 10
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(
        self,
        train_loader,
        val_loader,
        num_epochs: int,
        verbose: bool = True
    ):
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.scheduler.step(val_loss)
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                self.patience_counter += 1
            
            if verbose:
                print(f'Epoch {epoch+1}/{num_epochs}:')
                print(f'  Train Loss: {train_loss:.4f}')
                print(f'  Val Loss: {val_loss:.4f}')
                print(f'  Learning Rate: {self.optimizer.param_groups[0]["lr"]:.6f}')
            
            if self.patience_counter >= self.early_stopping_patience:
                print(f'Early stopping triggered at epoch {epoch+1}')
                break
        
        self.model.load_state_dict(torch.load('best_model.pth'))
```

### Model Deployment

```python
class ModelDeployment:
    def __init__(self, model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.load(model_path)
        self.model.to(self.device)
        self.model.eval()
        
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        input_tensor = torch.FloatTensor(input_data).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(input_tensor)
        
        return predictions.cpu().numpy()
    
    def batch_predict(
        self,
        input_data: np.ndarray,
        batch_size: int = 32
    ) -> np.ndarray:
        num_samples = len(input_data)
        predictions = []
        
        for i in range(0, num_samples, batch_size):
            batch = input_data[i:i+batch_size]
            batch_pred = self.predict(batch)
            predictions.append(batch_pred)
        
        return np.concatenate(predictions, axis=0)
```
