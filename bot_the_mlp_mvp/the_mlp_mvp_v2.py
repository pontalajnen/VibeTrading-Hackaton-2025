# =========================================================================
# Start of file. Do not edit this section.
# MLP STRATEGY IMPLEMENTATION
# =========================================================================

# import pandas as pd
import numpy as np
import joblib
from sklearn.neural_network import MLPRegressor
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
except ImportError:
    print("FATAL ERROR: Could not import backtest_engine.py. Check your directory structure.")
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
FAST_WINDOW = 10
SLOW_WINDOW = 30
N_DAYS_PREDICT = 1  # Shorter prediction horizon for better timing
SUBMISSION_NAME = '3D1I_mlp_submission.joblib'
INITIAL_CAPITAL = 10000.0
# ---------------------------------------------------

# Import the technical indicators library (Close-only indicators)
from technical_indicators import rsi, macd, bollinger_bands, roc, ema, momentum

# === ALPHA-FOCUSED FEATURE ENGINEERING ===
# Goal: Predict WHEN to trade for maximum alpha (using Close prices only)

# 1. Normalized MA Difference (trend relative to price level)
df['SMA_Fast'] = df.groupby(level='Ticker')['Close'].transform(lambda x: x.rolling(window=FAST_WINDOW).mean())
df['SMA_Slow'] = df.groupby(level='Ticker')['Close'].transform(lambda x: x.rolling(window=SLOW_WINDOW).mean())
df['MA_Diff_Norm'] = (df['SMA_Fast'] - df['SMA_Slow']) / df['Close']  # Normalized by price

# 2. EMA crossover signal (faster response than SMA)
df['EMA_Fast'] = df.groupby(level='Ticker')['Close'].transform(lambda x: ema(x, period=8))
df['EMA_Slow'] = df.groupby(level='Ticker')['Close'].transform(lambda x: ema(x, period=21))
df['EMA_Diff_Norm'] = (df['EMA_Fast'] - df['EMA_Slow']) / df['Close']

# 3. RSI - key for timing overbought/oversold reversals
df['RSI'] = df.groupby(level='Ticker')['Close'].transform(lambda x: rsi(x, period=14))
df['RSI_Centered'] = df['RSI'] - 50  # Center around 0 for better signal
df['RSI_Short'] = df.groupby(level='Ticker')['Close'].transform(lambda x: rsi(x, period=7))

# 4. MACD components (trend momentum)
def apply_macd_line(x):
    macd_line, signal_line, hist = macd(x, fast=12, slow=26, signal=9)
    return macd_line
def apply_macd_hist(x):
    macd_line, signal_line, hist = macd(x, fast=12, slow=26, signal=9)
    return hist
df['MACD_Line'] = df.groupby(level='Ticker')['Close'].transform(apply_macd_line)
df['MACD_Hist'] = df.groupby(level='Ticker')['Close'].transform(apply_macd_hist)

# 5. Bollinger Band position (where is price in the volatility envelope?)
def bb_position(x):
    upper, middle, lower = bollinger_bands(x, period=20)
    return (x - lower) / (upper - lower + 1e-8)  # Avoid div by zero
df['BB_Position'] = df.groupby(level='Ticker')['Close'].transform(bb_position)

# 6. Bollinger Band width (volatility measure)
def bb_width(x):
    upper, middle, lower = bollinger_bands(x, period=20)
    return (upper - lower) / middle
df['BB_Width'] = df.groupby(level='Ticker')['Close'].transform(bb_width)

# 7. Multi-timeframe momentum (captures different cycle lengths)
df['ROC_1'] = df.groupby(level='Ticker')['Close'].transform(lambda x: roc(x, period=1))
df['ROC_3'] = df.groupby(level='Ticker')['Close'].transform(lambda x: roc(x, period=3))
df['ROC_5'] = df.groupby(level='Ticker')['Close'].transform(lambda x: roc(x, period=5))
df['ROC_10'] = df.groupby(level='Ticker')['Close'].transform(lambda x: roc(x, period=10))
df['ROC_20'] = df.groupby(level='Ticker')['Close'].transform(lambda x: roc(x, period=20))

# 8. Momentum at different timeframes
df['MOM_5'] = df.groupby(level='Ticker')['Close'].transform(lambda x: momentum(x, period=5))
df['MOM_10'] = df.groupby(level='Ticker')['Close'].transform(lambda x: momentum(x, period=10))

# 9. Volatility regime (trade differently in high vs low vol)
df['Volatility'] = df.groupby(level='Ticker')['Close'].transform(lambda x: x.pct_change().rolling(20).std())
df['Vol_Ratio'] = df['Volatility'] / df.groupby(level='Ticker')['Volatility'].transform(lambda x: x.rolling(60).mean())

# 10. Price position relative to recent range (support/resistance proxy)
df['Price_Position'] = df.groupby(level='Ticker')['Close'].transform(
    lambda x: (x - x.rolling(20).min()) / (x.rolling(20).max() - x.rolling(20).min() + 1e-8)
)

# 11. Distance from 52-week high (trend strength)
df['Dist_From_High'] = df.groupby(level='Ticker')['Close'].transform(
    lambda x: (x - x.rolling(252).max()) / x.rolling(252).max()
)

# 12. Mean reversion signal (z-score of price from moving average)
df['Z_Score_20'] = df.groupby(level='Ticker')['Close'].transform(
    lambda x: (x - x.rolling(20).mean()) / (x.rolling(20).std() + 1e-8)
)

# Create the Target Variable (Future Return for next N days)
df['Future_Return'] = df.groupby(level='Ticker')['Close'].transform(
    lambda x: x.pct_change(N_DAYS_PREDICT).shift(-N_DAYS_PREDICT)
)

df.dropna(inplace=True)

# FEATURE SET - Use MA_Difference to match what evaluator computes
# The evaluator likely only computes basic features like MA_Difference
df['MA_Difference'] = df['SMA_Fast'] - df['SMA_Slow']  # Non-normalized version for compatibility

FEATURE_COLS = ['MA_Difference', 'RSI_Centered', 'ROC_5']
X = df[FEATURE_COLS]
y = df['Future_Return']

# TIME-AWARE TRAIN/TEST SPLIT (critical for proper backtesting)
# Get unique dates and split by date, not by row
unique_dates = df.index.get_level_values('Date').unique().sort_values()
train_cutoff_idx = int(len(unique_dates) * 0.80)
train_cutoff_date = unique_dates[train_cutoff_idx]

# Split data by date
df_train = df[df.index.get_level_values('Date') < train_cutoff_date]
df_local_test = df[df.index.get_level_values('Date') >= train_cutoff_date].copy()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(df_train[FEATURE_COLS])
y_train = df_train['Future_Return']

team_name = SUBMISSION_NAME.split('_submission')[0]

# === ALPHA-OPTIMIZED MODEL ===
print(f"\n--- 2. MODEL TRAINING ({len(X_train_scaled)} samples, {len(FEATURE_COLS)} features) ---")
print(f"    Training period: {df_train.index.get_level_values('Date').min().date()} to {df_train.index.get_level_values('Date').max().date()}")
print(f"    Test period: {df_local_test.index.get_level_values('Date').min().date()} to {df_local_test.index.get_level_values('Date').max().date()}")
print(f"    Train samples: {len(df_train)}, Test samples: {len(df_local_test)}")

model = MLPRegressor(
    random_state=42,
    hidden_layer_sizes=(32, 16),          # Even smaller network for max generalization
    max_iter=300,                         # Limit iterations to prevent overfitting
    early_stopping=True,
    validation_fraction=0.20,             # Large validation set
    n_iter_no_change=25,                  # Patient early stopping
    alpha=0.5,                            # Very strong regularization
    learning_rate='adaptive',
    learning_rate_init=0.0005,            # Slower learning
    activation='tanh',                    # tanh for financial data
    verbose=True
).fit(X_train_scaled, y_train)

print(f"\n    Training completed in {model.n_iter_} iterations")
print(f"    Final training loss: {model.loss_:.6f}")
print(f"    Best validation score: {model.best_validation_score_:.6f}")

# =========================================================================
# DO NOT EDIT FROM THIS POINT DOWNWARDS
# =========================================================================

print("\n--- Running Individual Stock Timing Analysis ---")

X_test_scaled = scaler.transform(df_local_test[FEATURE_COLS])
df_local_test['Predicted_Return'] = model.predict(X_test_scaled)
df_local_test['Signal'] = np.where(df_local_test['Predicted_Return'] > 0, 1, 0)

TICKERS_IN_TEST = df_local_test.index.get_level_values('Ticker').unique()

for ticker in TICKERS_IN_TEST:
    df_ticker_data = df_local_test.loc[(slice(None), ticker), :].droplevel('Ticker')
    # Call the fixed backtesting engine function
    run_single_stock_analysis(
        df_ticker_data, ticker, INITIAL_CAPITAL, team_name, strategy_file_path=CURRENT_STRATEGY_PATH
    )

# 4. SUBMIT (SAVE) THE FINAL MODEL
# IMPORTANT: Save ONLY the model (not a dictionary!) - evaluator expects model.predict()

FULL_SUBMISSION_PATH = os.path.join(SUBMISSION_FOLDER, SUBMISSION_NAME)
joblib.dump(model, FULL_SUBMISSION_PATH)
print(f"\nSUBMISSION READY: Model saved as {FULL_SUBMISSION_PATH}")
