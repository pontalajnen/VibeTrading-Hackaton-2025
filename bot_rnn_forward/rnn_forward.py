# =========================================================================
# Start of file. Do not edit this section.
# RNN + FEEDFORWARD STRATEGY IMPLEMENTATION
# =========================================================================

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
CURRENT_STRATEGY_PATH = os.path.abspath(__file__)

try:
    from backtest_engine import run_single_stock_analysis
    from data.load_data import load_training_data
    from technical_indicators import sma, ema, rsi, macd, bollinger_bands, momentum, roc
except ImportError as e:
    print(f"FATAL ERROR: Could not import required modules. Check your directory structure. {e}")
    exit()

df = load_training_data()
if df.empty:
    print("Cannot proceed without data. Exiting.")
    exit()

SUBMISSION_FOLDER = os.path.join(project_root, 'submissions')

# Ensure the submission directory exists
if not os.path.exists(SUBMISSION_FOLDER):
    os.makedirs(SUBMISSION_FOLDER)

# =========================================================================
# MODIFY FROM THIS POINT DOWNWARDS
# =========================================================================
# --- CONFIGURATION (Participants can adjust these) ---
SEQUENCE_LENGTH = 20         # Number of days to look back for RNN
N_DAYS_PREDICT = 2           # Days ahead to predict
SUBMISSION_NAME = 'my_team_name_rnn_submission.joblib'
INITIAL_CAPITAL = 10000.0

# RNN Hyperparameters
HIDDEN_SIZE = 64             # LSTM hidden size
NUM_LAYERS = 2               # Number of LSTM layers
FEEDFORWARD_SIZE = 32        # Size of feedforward layer after RNN
DROPOUT = 0.2                # Dropout rate
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 64
# ---------------------------------------------------

print("\n--- 1. FEATURE ENGINEERING ---")

# ============================================================
# CREATE ALL AVAILABLE FEATURES (only using Close price data)
# ============================================================

# Simple Moving Averages (multiple periods)
for period in [5, 10, 20, 50]:
    df[f'SMA_{period}'] = df.groupby(level='Ticker')['Close'].transform(lambda x: sma(x, period=period))

# Exponential Moving Averages (multiple periods)
for period in [5, 10, 20, 50]:
    df[f'EMA_{period}'] = df.groupby(level='Ticker')['Close'].transform(lambda x: ema(x, period=period))

# RSI (Relative Strength Index)
df['RSI_14'] = df.groupby(level='Ticker')['Close'].transform(lambda x: rsi(x, period=14))

# MACD (Moving Average Convergence Divergence)
def compute_macd(series):
    macd_line, signal_line, histogram = macd(series, fast=12, slow=26, signal=9)
    return pd.DataFrame({
        'MACD_Line': macd_line,
        'MACD_Signal': signal_line,
        'MACD_Histogram': histogram
    }, index=series.index)

macd_df = df.groupby(level='Ticker')['Close'].apply(compute_macd)
# Handle MultiIndex properly
if isinstance(macd_df.index, pd.MultiIndex) and macd_df.index.nlevels > 2:
    macd_df = macd_df.droplevel(0)
df = df.join(macd_df)

# Bollinger Bands
def compute_bollinger(series):
    upper, middle, lower = bollinger_bands(series, period=20, std_dev=2.0)
    # Create normalized position within bands (0 = at lower band, 1 = at upper band)
    bb_position = (series - lower) / (upper - lower)
    bb_width = (upper - lower) / middle  # Normalized band width
    return pd.DataFrame({
        'BB_Upper': upper,
        'BB_Middle': middle,
        'BB_Lower': lower,
        'BB_Position': bb_position,
        'BB_Width': bb_width
    }, index=series.index)

bb_df = df.groupby(level='Ticker')['Close'].apply(compute_bollinger)
if isinstance(bb_df.index, pd.MultiIndex) and bb_df.index.nlevels > 2:
    bb_df = bb_df.droplevel(0)
df = df.join(bb_df)

# Momentum
for period in [5, 10, 20]:
    df[f'Momentum_{period}'] = df.groupby(level='Ticker')['Close'].transform(lambda x: momentum(x, period=period))

# Rate of Change (ROC)
for period in [5, 10, 20]:
    df[f'ROC_{period}'] = df.groupby(level='Ticker')['Close'].transform(lambda x: roc(x, period=period))

# Price relative to moving averages (normalized)
df['Price_to_SMA20'] = df['Close'] / df['SMA_20'] - 1
df['Price_to_SMA50'] = df['Close'] / df['SMA_50'] - 1
df['Price_to_EMA20'] = df['Close'] / df['EMA_20'] - 1

# MA Crossover signals
df['SMA_Fast_Slow_Diff'] = df['SMA_10'] - df['SMA_50']
df['EMA_Fast_Slow_Diff'] = df['EMA_10'] - df['EMA_50']

# Daily returns (for additional context)
df['Daily_Return'] = df.groupby(level='Ticker')['Close'].transform(lambda x: x.pct_change())

# Volatility (rolling std of returns)
df['Volatility_10'] = df.groupby(level='Ticker')['Daily_Return'].transform(lambda x: x.rolling(10).std())
df['Volatility_20'] = df.groupby(level='Ticker')['Daily_Return'].transform(lambda x: x.rolling(20).std())

# Create the Target Variable (Future Return)
df['Future_Return'] = df.groupby(level='Ticker')['Close'].transform(
    lambda x: x.pct_change(N_DAYS_PREDICT).shift(-N_DAYS_PREDICT)
)

# Drop NaN values
df.dropna(inplace=True)

# ============================================================
# DEFINE FEATURE COLUMNS (all available Close-based features)
# ============================================================
FEATURE_COLS = [
    # SMAs
    'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50',
    # EMAs
    'EMA_5', 'EMA_10', 'EMA_20', 'EMA_50',
    # RSI
    'RSI_14',
    # MACD
    'MACD_Line', 'MACD_Signal', 'MACD_Histogram',
    # Bollinger Bands
    'BB_Position', 'BB_Width',
    # Momentum
    'Momentum_5', 'Momentum_10', 'Momentum_20',
    # ROC
    'ROC_5', 'ROC_10', 'ROC_20',
    # Price ratios
    'Price_to_SMA20', 'Price_to_SMA50', 'Price_to_EMA20',
    # MA differences
    'SMA_Fast_Slow_Diff', 'EMA_Fast_Slow_Diff',
    # Returns & Volatility
    'Daily_Return', 'Volatility_10', 'Volatility_20'
]

print(f"Total features: {len(FEATURE_COLS)}")
print(f"Features: {FEATURE_COLS}")

# ============================================================
# RNN MODEL DEFINITION
# ============================================================

class RNNForwardNetwork(nn.Module):
    """
    RNN (LSTM) followed by feedforward layers for time series prediction.
    """
    def __init__(self, input_size, hidden_size, num_layers, ff_size, dropout=0.2):
        super(RNNForwardNetwork, self).__init__()
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Feedforward layers after RNN
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_size, ff_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, ff_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size // 2, 1)
        )
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Use the last hidden state
        last_hidden = lstm_out[:, -1, :]
        # Pass through feedforward network
        output = self.feedforward(last_hidden)
        return output.squeeze(-1)


def create_sequences(data, target, seq_length):
    """
    Create sequences for RNN training.
    
    Args:
        data: Feature array (n_samples, n_features)
        target: Target array (n_samples,)
        seq_length: Number of time steps to look back
        
    Returns:
        X_seq: Sequences (n_sequences, seq_length, n_features)
        y_seq: Targets for each sequence
    """
    X_seq, y_seq = [], []
    for i in range(len(data) - seq_length):
        X_seq.append(data[i:i + seq_length])
        y_seq.append(target[i + seq_length])
    return np.array(X_seq), np.array(y_seq)


# ============================================================
# PREPARE DATA PER TICKER (to maintain temporal order)
# ============================================================

print("\n--- 2. DATA PREPARATION ---")

# Split data by ticker to maintain temporal order within each stock
train_sequences = []
train_targets = []
test_data_dict = {}

train_size_ratio = 0.80

for ticker in df.index.get_level_values('Ticker').unique():
    ticker_df = df.loc[(slice(None), ticker), :].droplevel('Ticker')
    
    # Split into train and test
    train_size = int(len(ticker_df) * train_size_ratio)
    
    train_df = ticker_df.iloc[:train_size]
    test_df = ticker_df.iloc[train_size:]
    
    # Scale features per ticker
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[FEATURE_COLS])
    y_train = train_df['Future_Return'].values
    
    X_test = scaler.transform(test_df[FEATURE_COLS])
    y_test = test_df['Future_Return'].values
    
    # Create sequences for training
    X_seq, y_seq = create_sequences(X_train, y_train, SEQUENCE_LENGTH)
    train_sequences.append(X_seq)
    train_targets.append(y_seq)
    
    # Store test data for later evaluation
    test_data_dict[ticker] = {
        'df': test_df,
        'X_test': X_test,
        'y_test': y_test,
        'scaler': scaler
    }
    
    print(f"  {ticker}: {len(X_seq)} train sequences, {len(test_df)} test samples")

# Combine all training sequences
X_train_all = np.concatenate(train_sequences, axis=0)
y_train_all = np.concatenate(train_targets, axis=0)

print(f"\nTotal training sequences: {len(X_train_all)}")
print(f"Sequence shape: {X_train_all.shape}")

# ============================================================
# TRAIN RNN MODEL
# ============================================================

print(f"\n--- 3. MODEL TRAINING ({len(X_train_all)} sequences) ---")

# Convert to PyTorch tensors
X_tensor = torch.FloatTensor(X_train_all)
y_tensor = torch.FloatTensor(y_train_all)

# Create DataLoader
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = RNNForwardNetwork(
    input_size=len(FEATURE_COLS),
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    ff_size=FEEDFORWARD_SIZE,
    dropout=DROPOUT
).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Training loop
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for batch_X, batch_y in dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward()
        
        # Gradient clipping for RNN stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    scheduler.step(avg_loss)
    
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.6f}")

print("Training complete!")

# =========================================================================
# DO NOT EDIT FROM THIS POINT DOWNWARDS
# =========================================================================

print("\n--- Running Individual Stock Timing Analysis ---")

team_name = SUBMISSION_NAME.split('_submission')[0]
model.eval()

for ticker, data in test_data_dict.items():
    test_df = data['df'].copy()
    X_test = data['X_test']
    
    # Create sequences for test data
    if len(X_test) >= SEQUENCE_LENGTH:
        X_test_seq, _ = create_sequences(X_test, np.zeros(len(X_test)), SEQUENCE_LENGTH)
        
        # Predict
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test_seq).to(device)
            predictions = model(X_test_tensor).cpu().numpy()
        
        # Align predictions with test DataFrame (skip first SEQUENCE_LENGTH rows)
        test_df = test_df.iloc[SEQUENCE_LENGTH:].copy()
        test_df['Predicted_Return'] = predictions
        test_df['Signal'] = np.where(test_df['Predicted_Return'] > 0, 1, 0)
        
        # Run backtest
        run_single_stock_analysis(
            test_df, ticker, INITIAL_CAPITAL, team_name, strategy_file_path=CURRENT_STRATEGY_PATH
        )
    else:
        print(f"  Skipping {ticker}: not enough test data for sequence length {SEQUENCE_LENGTH}")

# 4. SUBMIT (SAVE) THE FINAL MODEL

FULL_SUBMISSION_PATH = os.path.join(SUBMISSION_FOLDER, SUBMISSION_NAME)
joblib.dump(model, FULL_SUBMISSION_PATH)
print(f"\nSUBMISSION READY: Model saved as {FULL_SUBMISSION_PATH}")

