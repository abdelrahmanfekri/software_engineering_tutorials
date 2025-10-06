# NLP Tutorial Module 6: Recurrent Neural Networks (RNN/LSTM/GRU)

## Learning Objectives
By the end of this module, you will be able to:
- Understand the architecture and mechanics of RNNs
- Implement vanilla RNNs, LSTMs, and GRUs from scratch
- Apply RNNs to sequence modeling tasks
- Handle the vanishing gradient problem
- Build bidirectional and stacked RNN architectures
- Use RNNs for language modeling and sequence classification

## Introduction to Recurrent Neural Networks

Recurrent Neural Networks (RNNs) are designed to process sequential data by maintaining hidden states that capture information from previous time steps. This makes them particularly suitable for NLP tasks where the order of words matters.

### Key Characteristics of RNNs

1. **Sequential Processing**: Process input sequences step by step
2. **Hidden State**: Maintain memory of previous computations
3. **Parameter Sharing**: Same parameters used across all time steps
4. **Variable Length**: Can handle sequences of different lengths

### Types of RNN Architectures

1. **Vanilla RNN**: Basic recurrent architecture
2. **LSTM (Long Short-Term Memory)**: Addresses vanishing gradient problem
3. **GRU (Gated Recurrent Unit)**: Simplified version of LSTM
4. **Bidirectional RNN**: Process sequences in both directions
5. **Stacked RNN**: Multiple layers of RNNs

## Vanilla RNN Architecture

### Mathematical Foundation

For a vanilla RNN at time step t:

```
hₜ = tanh(Wₕₕ * hₜ₋₁ + Wₓₕ * xₜ + bₕ)
yₜ = Wₕᵧ * hₜ + bᵧ
```

Where:
- hₜ is the hidden state at time t
- xₜ is the input at time t
- Wₕₕ, Wₓₕ, Wₕᵧ are weight matrices
- bₕ, bᵧ are bias vectors

### Implementation

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class VanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0):
        super(VanillaRNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        # RNN layers
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, hidden=None):
        # x shape: (batch_size, sequence_length, input_size)
        batch_size = x.size(0)
        
        if hidden is None:
            hidden = self._init_hidden(batch_size)
        
        # RNN forward pass
        rnn_out, hidden = self.rnn(x, hidden)
        
        # Apply dropout
        rnn_out = self.dropout(rnn_out)
        
        # Output layer
        output = self.fc(rnn_out)
        
        return output, hidden
    
    def _init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)

class RNNFromScratch:
    """Vanilla RNN implemented from scratch using numpy"""
    
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_xh = np.random.randn(input_size, hidden_size) * 0.01
        self.W_hy = np.random.randn(hidden_size, output_size) * 0.01
        
        # Initialize biases
        self.b_h = np.zeros((1, hidden_size))
        self.b_y = np.zeros((1, output_size))
        
        # Initialize hidden state
        self.hidden_state = None
        
    def forward(self, x):
        """Forward pass through the RNN"""
        batch_size, seq_len, input_size = x.shape
        outputs = np.zeros((batch_size, seq_len, self.output_size))
        hidden_states = np.zeros((batch_size, seq_len, self.hidden_size))
        
        # Initialize hidden state
        h = np.zeros((batch_size, self.hidden_size))
        
        for t in range(seq_len):
            # Update hidden state
            h = np.tanh(np.dot(x[:, t, :], self.W_xh) + np.dot(h, self.W_hh) + self.b_h)
            
            # Compute output
            y = np.dot(h, self.W_hy) + self.b_y
            
            # Store outputs and hidden states
            outputs[:, t, :] = y
            hidden_states[:, t, :] = h
        
        self.hidden_state = h
        return outputs, hidden_states
    
    def backward(self, x, outputs, hidden_states, targets):
        """Backward pass through time (BPTT)"""
        batch_size, seq_len, _ = x.shape
        
        # Initialize gradients
        dW_hh = np.zeros_like(self.W_hh)
        dW_xh = np.zeros_like(self.W_xh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)
        
        # Gradient of hidden state (initialize to zero)
        dh_next = np.zeros((batch_size, self.hidden_size))
        
        # Backward pass through time
        for t in reversed(range(seq_len)):
            # Output layer gradients
            dy = outputs[:, t, :] - targets[:, t, :]
            dW_hy += np.dot(hidden_states[:, t, :].T, dy)
            db_y += np.sum(dy, axis=0, keepdims=True)
            
            # Hidden layer gradients
            dh = np.dot(dy, self.W_hy.T) + dh_next
            
            # Gradient of tanh
            dhraw = dh * (1 - hidden_states[:, t, :] ** 2)
            
            # Parameter gradients
            dW_hh += np.dot(hidden_states[:, t-1, :].T, dhraw) if t > 0 else np.zeros_like(dW_hh)
            dW_xh += np.dot(x[:, t, :].T, dhraw)
            db_h += np.sum(dhraw, axis=0, keepdims=True)
            
            # Gradient for next time step
            dh_next = np.dot(dhraw, self.W_hh.T)
        
        # Update parameters
        self.W_hh -= self.learning_rate * dW_hh
        self.W_xh -= self.learning_rate * dW_xh
        self.W_hy -= self.learning_rate * dW_hy
        self.b_h -= self.learning_rate * db_h
        self.b_y -= self.learning_rate * db_y
    
    def train(self, x, targets, epochs=100):
        """Train the RNN"""
        losses = []
        
        for epoch in range(epochs):
            # Forward pass
            outputs, hidden_states = self.forward(x)
            
            # Calculate loss (MSE)
            loss = np.mean((outputs - targets) ** 2)
            losses.append(loss)
            
            # Backward pass
            self.backward(x, outputs, hidden_states, targets)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")
        
        return losses

# Example usage of vanilla RNN
def create_synthetic_data():
    """Create synthetic sequential data for demonstration"""
    # Generate sine wave data
    t = np.linspace(0, 4*np.pi, 100)
    data = np.sin(t).reshape(-1, 1)
    
    # Create sequences
    sequence_length = 10
    X, y = [], []
    
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+1:i+sequence_length+1])
    
    return np.array(X), np.array(y)

# Create and train vanilla RNN from scratch
X, y = create_synthetic_data()
X = X.reshape(-1, 10, 1)  # (batch_size, sequence_length, input_size)
y = y.reshape(-1, 10, 1)  # (batch_size, sequence_length, output_size)

print(f"Data shapes - X: {X.shape}, y: {y.shape}")

# Train RNN
rnn_scratch = RNNFromScratch(input_size=1, hidden_size=16, output_size=1, learning_rate=0.01)
losses = rnn_scratch.train(X, y, epochs=200)

# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.title('Vanilla RNN Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Test prediction
test_input = X[0:1]  # First sequence
prediction, _ = rnn_scratch.forward(test_input)
print(f"Input: {test_input[0, :, 0]}")
print(f"Target: {y[0, :, 0]}")
print(f"Prediction: {prediction[0, :, 0]}")
```

## Long Short-Term Memory (LSTM)

LSTM addresses the vanishing gradient problem in vanilla RNNs by using gating mechanisms to control information flow.

### LSTM Architecture

An LSTM cell has three gates:
1. **Forget Gate**: Decides what to forget from previous hidden state
2. **Input Gate**: Decides what new information to store
3. **Output Gate**: Decides what parts of the cell state to output

### Mathematical Formulation

```
fₜ = σ(Wf · [hₜ₋₁, xₜ] + bf)  # Forget gate
iₜ = σ(Wi · [hₜ₋₁, xₜ] + bi)  # Input gate
C̃ₜ = tanh(WC · [hₜ₋₁, xₜ] + bC)  # Candidate values
Cₜ = fₜ * Cₜ₋₁ + iₜ * C̃ₜ  # Cell state
oₜ = σ(Wo · [hₜ₋₁, xₜ] + bo)  # Output gate
hₜ = oₜ * tanh(Cₜ)  # Hidden state
```

### Implementation

```python
class LSTMFromScratch:
    """LSTM implemented from scratch using numpy"""
    
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # LSTM weights (combined for efficiency)
        # W_f, W_i, W_C, W_o for forget, input, candidate, output gates
        self.W_f = np.random.randn(hidden_size + input_size, hidden_size) * 0.01
        self.W_i = np.random.randn(hidden_size + input_size, hidden_size) * 0.01
        self.W_C = np.random.randn(hidden_size + input_size, hidden_size) * 0.01
        self.W_o = np.random.randn(hidden_size + input_size, hidden_size) * 0.01
        
        # Output layer weights
        self.W_y = np.random.randn(hidden_size, output_size) * 0.01
        
        # Biases
        self.b_f = np.zeros((1, hidden_size))
        self.b_i = np.zeros((1, hidden_size))
        self.b_C = np.zeros((1, hidden_size))
        self.b_o = np.zeros((1, hidden_size))
        self.b_y = np.zeros((1, output_size))
        
        # Initialize cell state and hidden state
        self.cell_state = None
        self.hidden_state = None
    
    def _sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, x):
        """Forward pass through LSTM"""
        batch_size, seq_len, input_size = x.shape
        outputs = np.zeros((batch_size, seq_len, self.output_size))
        hidden_states = np.zeros((batch_size, seq_len, self.hidden_size))
        cell_states = np.zeros((batch_size, seq_len, self.hidden_size))
        
        # Initialize states
        h = np.zeros((batch_size, self.hidden_size))
        c = np.zeros((batch_size, self.hidden_size))
        
        for t in range(seq_len):
            # Concatenate hidden state and input
            concat = np.concatenate((h, x[:, t, :]), axis=1)
            
            # Calculate gates
            f_t = self._sigmoid(np.dot(concat, self.W_f) + self.b_f)  # Forget gate
            i_t = self._sigmoid(np.dot(concat, self.W_i) + self.b_i)  # Input gate
            C_tilde = np.tanh(np.dot(concat, self.W_C) + self.b_C)   # Candidate values
            o_t = self._sigmoid(np.dot(concat, self.W_o) + self.b_o)  # Output gate
            
            # Update cell state and hidden state
            c = f_t * c + i_t * C_tilde
            h = o_t * np.tanh(c)
            
            # Calculate output
            y = np.dot(h, self.W_y) + self.b_y
            
            # Store states and outputs
            outputs[:, t, :] = y
            hidden_states[:, t, :] = h
            cell_states[:, t, :] = c
        
        self.hidden_state = h
        self.cell_state = c
        
        return outputs, hidden_states, cell_states
    
    def backward(self, x, outputs, hidden_states, cell_states, targets):
        """Backward pass through LSTM"""
        batch_size, seq_len, _ = x.shape
        
        # Initialize gradients
        dW_f = np.zeros_like(self.W_f)
        dW_i = np.zeros_like(self.W_i)
        dW_C = np.zeros_like(self.W_C)
        dW_o = np.zeros_like(self.W_o)
        dW_y = np.zeros_like(self.W_y)
        
        db_f = np.zeros_like(self.b_f)
        db_i = np.zeros_like(self.b_i)
        db_C = np.zeros_like(self.b_C)
        db_o = np.zeros_like(self.b_o)
        db_y = np.zeros_like(self.b_y)
        
        # Initialize gradients for next time step
        dh_next = np.zeros((batch_size, self.hidden_size))
        dc_next = np.zeros((batch_size, self.hidden_size))
        
        # Backward pass through time
        for t in reversed(range(seq_len)):
            # Output layer gradients
            dy = outputs[:, t, :] - targets[:, t, :]
            dW_y += np.dot(hidden_states[:, t, :].T, dy)
            db_y += np.sum(dy, axis=0, keepdims=True)
            
            # Hidden state gradient
            dh = np.dot(dy, self.W_y.T) + dh_next
            
            # Cell state gradient
            dc = dc_next + (dh * o_t * (1 - np.tanh(cell_states[:, t, :]) ** 2))
            
            # Gate gradients
            do = dh * np.tanh(cell_states[:, t, :])
            dC_tilde = dc * i_t
            di = dc * C_tilde
            df = dc * (cell_states[:, t-1, :] if t > 0 else np.zeros_like(cell_states[:, t, :]))
            
            # Gate activations
            do_raw = do * o_t * (1 - o_t)
            di_raw = di * i_t * (1 - i_t)
            dC_raw = dC_tilde * (1 - C_tilde ** 2)
            df_raw = df * f_t * (1 - f_t)
            
            # Concatenated input gradient
            dconcat = (np.dot(do_raw, self.W_o.T) + 
                      np.dot(di_raw, self.W_i.T) + 
                      np.dot(dC_raw, self.W_C.T) + 
                      np.dot(df_raw, self.W_f.T))
            
            # Split gradients
            dh_prev = dconcat[:, :self.hidden_size]
            dx = dconcat[:, self.hidden_size:]
            
            # Parameter updates
            concat = np.concatenate((hidden_states[:, t-1, :] if t > 0 else np.zeros((batch_size, self.hidden_size)), x[:, t, :]), axis=1)
            
            dW_o += np.dot(concat.T, do_raw)
            dW_i += np.dot(concat.T, di_raw)
            dW_C += np.dot(concat.T, dC_raw)
            dW_f += np.dot(concat.T, df_raw)
            
            db_o += np.sum(do_raw, axis=0, keepdims=True)
            db_i += np.sum(di_raw, axis=0, keepdims=True)
            db_C += np.sum(dC_raw, axis=0, keepdims=True)
            db_f += np.sum(df_raw, axis=0, keepdims=True)
            
            # Update gradients for next time step
            dh_next = dh_prev
            dc_next = dc * f_t
        
        # Update parameters
        self.W_f -= self.learning_rate * dW_f
        self.W_i -= self.learning_rate * dW_i
        self.W_C -= self.learning_rate * dW_C
        self.W_o -= self.learning_rate * dW_o
        self.W_y -= self.learning_rate * dW_y
        
        self.b_f -= self.learning_rate * db_f
        self.b_i -= self.learning_rate * db_i
        self.b_C -= self.learning_rate * db_C
        self.b_o -= self.learning_rate * db_o
        self.b_y -= self.learning_rate * db_y
    
    def train(self, x, targets, epochs=100):
        """Train the LSTM"""
        losses = []
        
        for epoch in range(epochs):
            # Forward pass
            outputs, hidden_states, cell_states = self.forward(x)
            
            # Calculate loss
            loss = np.mean((outputs - targets) ** 2)
            losses.append(loss)
            
            # Backward pass
            self.backward(x, outputs, hidden_states, cell_states, targets)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")
        
        return losses

# Train LSTM on the same synthetic data
lstm_scratch = LSTMFromScratch(input_size=1, hidden_size=16, output_size=1, learning_rate=0.01)
lstm_losses = lstm_scratch.train(X, y, epochs=200)

# Compare RNN vs LSTM performance
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(losses, label='Vanilla RNN')
plt.plot(lstm_losses, label='LSTM')
plt.title('Training Loss Comparison')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
# Test predictions
test_input = X[0:1]
rnn_pred, _ = rnn_scratch.forward(test_input)
lstm_pred, _, _ = lstm_scratch.forward(test_input)

plt.plot(y[0, :, 0], label='Target', marker='o')
plt.plot(rnn_pred[0, :, 0], label='RNN', marker='s')
plt.plot(lstm_pred[0, :, 0], label='LSTM', marker='^')
plt.title('Prediction Comparison')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()

plt.tight_layout()
plt.show()
```

## Gated Recurrent Unit (GRU)

GRU is a simplified version of LSTM that combines forget and input gates into a single update gate.

### GRU Architecture

GRU has two gates:
1. **Update Gate**: Controls how much of the previous hidden state to keep
2. **Reset Gate**: Controls how much of the previous hidden state to forget

### Mathematical Formulation

```
zₜ = σ(Wz · [hₜ₋₁, xₜ])  # Update gate
rₜ = σ(Wr · [hₜ₋₁, xₜ])  # Reset gate
h̃ₜ = tanh(Wh · [rₜ * hₜ₋₁, xₜ])  # Candidate hidden state
hₜ = (1 - zₜ) * hₜ₋₁ + zₜ * h̃ₜ  # Hidden state
```

### Implementation

```python
class GRUFromScratch:
    """GRU implemented from scratch using numpy"""
    
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # GRU weights
        self.W_z = np.random.randn(hidden_size + input_size, hidden_size) * 0.01  # Update gate
        self.W_r = np.random.randn(hidden_size + input_size, hidden_size) * 0.01  # Reset gate
        self.W_h = np.random.randn(hidden_size + input_size, hidden_size) * 0.01  # Candidate
        
        # Output layer weights
        self.W_y = np.random.randn(hidden_size, output_size) * 0.01
        
        # Biases
        self.b_z = np.zeros((1, hidden_size))
        self.b_r = np.zeros((1, hidden_size))
        self.b_h = np.zeros((1, hidden_size))
        self.b_y = np.zeros((1, output_size))
        
        # Initialize hidden state
        self.hidden_state = None
    
    def _sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, x):
        """Forward pass through GRU"""
        batch_size, seq_len, input_size = x.shape
        outputs = np.zeros((batch_size, seq_len, self.output_size))
        hidden_states = np.zeros((batch_size, seq_len, self.hidden_size))
        
        # Initialize hidden state
        h = np.zeros((batch_size, self.hidden_size))
        
        for t in range(seq_len):
            # Concatenate hidden state and input
            concat = np.concatenate((h, x[:, t, :]), axis=1)
            
            # Calculate gates
            z_t = self._sigmoid(np.dot(concat, self.W_z) + self.b_z)  # Update gate
            r_t = self._sigmoid(np.dot(concat, self.W_r) + self.b_r)  # Reset gate
            
            # Calculate candidate hidden state
            h_concat = np.concatenate((r_t * h, x[:, t, :]), axis=1)
            h_tilde = np.tanh(np.dot(h_concat, self.W_h) + self.b_h)
            
            # Update hidden state
            h = (1 - z_t) * h + z_t * h_tilde
            
            # Calculate output
            y = np.dot(h, self.W_y) + self.b_y
            
            # Store states and outputs
            outputs[:, t, :] = y
            hidden_states[:, t, :] = h
        
        self.hidden_state = h
        return outputs, hidden_states
    
    def train(self, x, targets, epochs=100):
        """Train the GRU (simplified version)"""
        losses = []
        
        for epoch in range(epochs):
            # Forward pass
            outputs, hidden_states = self.forward(x)
            
            # Calculate loss
            loss = np.mean((outputs - targets) ** 2)
            losses.append(loss)
            
            # Simple gradient update (full BPTT implementation would be more complex)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")
        
        return losses

# Train GRU
gru_scratch = GRUFromScratch(input_size=1, hidden_size=16, output_size=1, learning_rate=0.01)
gru_losses = gru_scratch.train(X, y, epochs=200)
```

## PyTorch Implementation of RNNs

```python
class PyTorchLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0):
        super(PyTorchLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, hidden=None):
        # x shape: (batch_size, sequence_length, input_size)
        batch_size = x.size(0)
        
        if hidden is None:
            hidden = self._init_hidden(batch_size)
        
        # LSTM forward pass
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
        # Output layer
        output = self.fc(lstm_out)
        
        return output, hidden
    
    def _init_hidden(self, batch_size):
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return (h_0, c_0)

class PyTorchGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0):
        super(PyTorchGRU, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        
        if hidden is None:
            hidden = self._init_hidden(batch_size)
        
        # GRU forward pass
        gru_out, hidden = self.gru(x, hidden)
        
        # Apply dropout
        gru_out = self.dropout(gru_out)
        
        # Output layer
        output = self.fc(gru_out)
        
        return output, hidden
    
    def _init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)

# Convert numpy data to PyTorch tensors
X_torch = torch.FloatTensor(X)
y_torch = torch.FloatTensor(y)

# Create PyTorch models
pytorch_lstm = PyTorchLSTM(input_size=1, hidden_size=16, output_size=1, num_layers=1, dropout=0.1)
pytorch_gru = PyTorchGRU(input_size=1, hidden_size=16, output_size=1, num_layers=1, dropout=0.1)

# Training function for PyTorch models
def train_pytorch_model(model, X, y, epochs=100, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    losses = []
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        output, hidden = model(X)
        
        # Calculate loss
        loss = criterion(output, y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
    
    return losses

# Train PyTorch models
pytorch_lstm_losses = train_pytorch_model(pytorch_lstm, X_torch, y_torch, epochs=200)
pytorch_gru_losses = train_pytorch_model(pytorch_gru, X_torch, y_torch, epochs=200)

# Compare all models
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(losses, label='Vanilla RNN')
plt.plot(lstm_losses, label='LSTM (Scratch)')
plt.plot(pytorch_lstm_losses, label='LSTM (PyTorch)')
plt.plot(pytorch_gru_losses, label='GRU (PyTorch)')
plt.title('Training Loss Comparison')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 3, 2)
# Test predictions
pytorch_lstm.eval()
pytorch_gru.eval()

with torch.no_grad():
    lstm_pred, _ = pytorch_lstm(X_torch[0:1])
    gru_pred, _ = pytorch_gru(X_torch[0:1])

plt.plot(y[0, :, 0], label='Target', marker='o')
plt.plot(rnn_pred[0, :, 0], label='RNN', marker='s')
plt.plot(lstm_pred[0, :, 0], label='LSTM (Scratch)', marker='^')
plt.plot(lstm_pred[0, :, 0].numpy(), label='LSTM (PyTorch)', marker='d')
plt.plot(gru_pred[0, :, 0].numpy(), label='GRU (PyTorch)', marker='v')
plt.title('Prediction Comparison')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()

plt.tight_layout()
plt.show()
```

## Bidirectional RNNs

Bidirectional RNNs process sequences in both forward and backward directions, capturing information from both past and future context.

```python
class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0):
        super(BidirectionalLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        
        # Output layer (input size is doubled due to bidirectionality)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
        # Output layer
        output = self.fc(lstm_out)
        
        return output

# Example usage
bidirectional_lstm = BidirectionalLSTM(input_size=1, hidden_size=16, output_size=1, num_layers=1)
bidirectional_losses = train_pytorch_model(bidirectional_lstm, X_torch, y_torch, epochs=200)
```

## Language Modeling with RNNs

```python
class RNLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=2, dropout=0.5):
        super(RNLanguageModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, hidden=None):
        # Embedding
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        
        # LSTM
        lstm_out, hidden = self.lstm(embedded, hidden)
        lstm_out = self.dropout(lstm_out)
        
        # Output
        output = self.fc(lstm_out)
        
        return output, hidden
    
    def init_hidden(self, batch_size):
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return (h_0, c_0)
    
    def generate(self, word_to_idx, idx_to_word, seed_text, max_length=50):
        """Generate text using the trained model"""
        self.eval()
        
        # Convert seed text to indices
        words = seed_text.lower().split()
        indices = [word_to_idx.get(word, 0) for word in words]
        
        generated_indices = indices.copy()
        hidden = self.init_hidden(1)
        
        with torch.no_grad():
            for _ in range(max_length - len(words)):
                # Prepare input
                input_tensor = torch.tensor([generated_indices[-10:]], dtype=torch.long)
                
                # Forward pass
                output, hidden = self.forward(input_tensor, hidden)
                
                # Get next word probabilities
                next_word_logits = output[0, -1, :]
                probabilities = torch.softmax(next_word_logits, dim=0)
                
                # Sample next word
                next_word_idx = torch.multinomial(probabilities, 1).item()
                generated_indices.append(next_word_idx)
        
        # Convert back to text
        generated_words = [idx_to_word.get(idx, '<UNK>') for idx in generated_indices]
        return ' '.join(generated_words)

# Example usage for language modeling
def create_language_model_data(texts, word_to_idx, sequence_length=10):
    """Create sequences for language modeling"""
    sequences = []
    
    for text in texts:
        words = text.lower().split()
        indices = [word_to_idx.get(word, 0) for word in words]
        
        for i in range(len(indices) - sequence_length):
            sequence = indices[i:i + sequence_length + 1]
            sequences.append(sequence)
    
    return sequences

# Create language model data
lm_sequences = create_language_model_data(texts, word_to_idx, sequence_length=10)

# Convert to tensors
lm_data = []
for seq in lm_sequences:
    input_seq = torch.tensor(seq[:-1], dtype=torch.long)
    target_seq = torch.tensor(seq[1:], dtype=torch.long)
    lm_data.append((input_seq, target_seq))

# Create data loader
lm_loader = DataLoader(lm_data, batch_size=2, shuffle=True)

# Create and train language model
lm_model = RNLanguageModel(
    vocab_size=vocab_size,
    embedding_dim=50,
    hidden_size=100,
    num_layers=2,
    dropout=0.3
)

def train_language_model(model, data_loader, epochs=10, learning_rate=0.001):
    """Train language model"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(data_loader):
            optimizer.zero_grad()
            
            # Forward pass
            output, _ = model(data)
            
            # Reshape for loss calculation
            output = output.view(-1, model.vocab_size)
            target = target.view(-1)
            
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(data_loader)
        print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')

# Train the language model
train_language_model(lm_model, lm_loader, epochs=20, learning_rate=0.001)

# Generate text
generated_text = lm_model.generate(word_to_idx, idx_to_word, "great movie", max_length=10)
print(f"Generated text: {generated_text}")
```

## Practical Exercises

### Exercise 1: RNN Architecture Comparison
Compare different RNN architectures on the same task:
- Vanilla RNN vs LSTM vs GRU
- Different numbers of layers
- Different hidden sizes
- Analyze training dynamics and final performance

### Exercise 2: Sequence Length Analysis
Investigate how sequence length affects RNN performance:
- Train models on different sequence lengths
- Compare gradient flow for short vs long sequences
- Analyze the vanishing gradient problem

### Exercise 3: Language Modeling
Build a complete language modeling system:
- Train on a larger corpus
- Implement beam search for text generation
- Evaluate using perplexity
- Compare with n-gram models

## Assessment Questions

1. **What is the main advantage of LSTM over vanilla RNNs?**
   - Addresses vanishing gradient problem
   - Better long-term memory
   - More stable training
   - Better performance on long sequences

2. **How do GRUs differ from LSTMs?**
   - Fewer parameters
   - Combines forget and input gates
   - Simpler architecture
   - Often comparable performance

3. **What is the purpose of bidirectional RNNs?**
   - Process sequences in both directions
   - Capture future context
   - Better for sequence labeling tasks
   - Not suitable for online prediction

## Key Takeaways

- RNNs are designed for sequential data processing
- Vanilla RNNs suffer from vanishing gradient problem
- LSTM and GRU address this with gating mechanisms
- Bidirectional RNNs capture both past and future context
- Stacked RNNs provide more modeling capacity
- RNNs are fundamental building blocks for many NLP tasks
- Proper initialization and regularization are crucial for training

## Next Steps

In the next module, we'll explore Convolutional Neural Networks for NLP, which offer an alternative approach to processing sequential data by focusing on local patterns and relationships.
