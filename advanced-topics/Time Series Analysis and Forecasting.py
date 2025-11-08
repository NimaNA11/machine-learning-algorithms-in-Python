"""
Time Series Analysis and Forecasting
Methods for analyzing and predicting sequential data.
Note: Install statsmodels with: pip install statsmodels
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Try to import statsmodels (optional)
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.seasonal import seasonal_decompose
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("statsmodels not installed. Install with: pip install statsmodels")
    print("Some time series methods will not be available.\n")

def generate_time_series():
    """Generate synthetic time series data"""
    np.random.seed(42)
    n = 200
    
    # Trend
    trend = np.linspace(0, 10, n)
    
    # Seasonality
    seasonality = 3 * np.sin(np.linspace(0, 4 * np.pi, n))
    
    # Noise
    noise = np.random.normal(0, 1, n)
    
    # Combine
    ts = trend + seasonality + noise
    
    return ts

def moving_average_forecast():
    """Simple Moving Average"""
    print("=" * 50)
    print("MOVING AVERAGE FORECAST")
    print("=" * 50)
    
    ts = generate_time_series()
    
    # Split data
    train_size = int(len(ts) * 0.8)
    train, test = ts[:train_size], ts[train_size:]
    
    # Moving average with different windows
    windows = [5, 10, 20]
    
    plt.figure(figsize=(15, 5))
    plt.plot(range(len(ts)), ts, label='Original', alpha=0.5)
    plt.axvline(x=train_size, color='r', linestyle='--', label='Train/Test Split')
    
    for window in windows:
        ma = np.convolve(train, np.ones(window)/window, mode='valid')
        # Forecast using last MA value
        forecast = np.full(len(test), ma[-1])
        
        rmse = np.sqrt(mean_squared_error(test, forecast))
        print(f"Window={window}: RMSE={rmse:.4f}")
        
        # Plot
        ma_full = np.concatenate([np.full(window-1, np.nan), ma])
        plt.plot(range(len(ma_full)), ma_full, label=f'MA-{window}', alpha=0.7)
    
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Moving Average Forecasting')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('moving_average.png')
    print("Plot saved as 'moving_average.png'\n")
    
    return ts

def exponential_smoothing():
    """Exponential Smoothing"""
    print("=" * 50)
    print("EXPONENTIAL SMOOTHING")
    print("=" * 50)
    
    ts = generate_time_series()
    train_size = int(len(ts) * 0.8)
    train, test = ts[:train_size], ts[train_size:]
    
    # Simple Exponential Smoothing
    alphas = [0.1, 0.3, 0.5, 0.9]
    
    plt.figure(figsize=(15, 5))
    plt.plot(range(len(ts)), ts, label='Original', alpha=0.5)
    plt.axvline(x=train_size, color='r', linestyle='--', label='Train/Test Split')
    
    for alpha in alphas:
        smoothed = np.zeros(len(train))
        smoothed[0] = train[0]
        
        for t in range(1, len(train)):
            smoothed[t] = alpha * train[t] + (1 - alpha) * smoothed[t-1]
        
        # Forecast
        forecast = np.full(len(test), smoothed[-1])
        rmse = np.sqrt(mean_squared_error(test, forecast))
        print(f"Alpha={alpha}: RMSE={rmse:.4f}")
        
        plt.plot(range(len(smoothed)), smoothed, label=f'α={alpha}', alpha=0.7)
    
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Exponential Smoothing')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('exponential_smoothing.png')
    print("Plot saved as 'exponential_smoothing.png'\n")

def demonstrate_arima():
    """ARIMA - AutoRegressive Integrated Moving Average"""
    if not STATSMODELS_AVAILABLE:
        print("ARIMA requires statsmodels. Skipping...\n")
        return
    
    print("=" * 50)
    print("ARIMA MODEL")
    print("=" * 50)
    
    ts = generate_time_series()
    train_size = int(len(ts) * 0.8)
    train, test = ts[:train_size], ts[train_size:]
    
    # Fit ARIMA(p,d,q) model
    # p: autoregressive order, d: differencing order, q: moving average order
    model = ARIMA(train, order=(2, 1, 2))
    fitted_model = model.fit()
    
    print(fitted_model.summary())
    
    # Forecast
    forecast = fitted_model.forecast(steps=len(test))
    
    # Evaluate
    rmse = np.sqrt(mean_squared_error(test, forecast))
    mae = mean_absolute_error(test, forecast)
    
    print(f"\nRMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    
    # Plot
    plt.figure(figsize=(15, 5))
    plt.plot(range(len(train)), train, label='Train', alpha=0.7)
    plt.plot(range(len(train), len(ts)), test, label='Test', alpha=0.7)
    plt.plot(range(len(train), len(ts)), forecast, label='Forecast', linestyle='--', alpha=0.7)
    plt.axvline(x=train_size, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('ARIMA Forecasting')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('arima_forecast.png')
    print("Plot saved as 'arima_forecast.png'\n")
    
    return fitted_model

def demonstrate_sarima():
    """SARIMA - Seasonal ARIMA"""
    if not STATSMODELS_AVAILABLE:
        print("SARIMA requires statsmodels. Skipping...\n")
        return
    
    print("=" * 50)
    print("SARIMA MODEL")
    print("=" * 50)
    
    ts = generate_time_series()
    train_size = int(len(ts) * 0.8)
    train, test = ts[:train_size], ts[train_size:]
    
    # Fit SARIMA(p,d,q)(P,D,Q,s) model
    # (P,D,Q,s): seasonal components with period s
    model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    fitted_model = model.fit(disp=False)
    
    # Forecast
    forecast = fitted_model.forecast(steps=len(test))
    
    # Evaluate
    rmse = np.sqrt(mean_squared_error(test, forecast))
    mae = mean_absolute_error(test, forecast)
    
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    
    # Plot
    plt.figure(figsize=(15, 5))
    plt.plot(range(len(train)), train, label='Train', alpha=0.7)
    plt.plot(range(len(train), len(ts)), test, label='Test', alpha=0.7)
    plt.plot(range(len(train), len(ts)), forecast, label='Forecast', linestyle='--', alpha=0.7)
    plt.axvline(x=train_size, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('SARIMA Forecasting')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('sarima_forecast.png')
    print("Plot saved as 'sarima_forecast.png'\n")
    
    return fitted_model

def demonstrate_hw_exponential_smoothing():
    """Holt-Winters Exponential Smoothing"""
    if not STATSMODELS_AVAILABLE:
        print("Holt-Winters requires statsmodels. Skipping...\n")
        return
    
    print("=" * 50)
    print("HOLT-WINTERS EXPONENTIAL SMOOTHING")
    print("=" * 50)
    
    ts = generate_time_series()
    train_size = int(len(ts) * 0.8)
    train, test = ts[:train_size], ts[train_size:]
    
    # Fit Holt-Winters model
    model = ExponentialSmoothing(
        train,
        seasonal_periods=12,
        trend='add',
        seasonal='add'
    )
    fitted_model = model.fit()
    
    # Forecast
    forecast = fitted_model.forecast(steps=len(test))
    
    # Evaluate
    rmse = np.sqrt(mean_squared_error(test, forecast))
    mae = mean_absolute_error(test, forecast)
    
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    
    # Plot
    plt.figure(figsize=(15, 5))
    plt.plot(range(len(train)), train, label='Train', alpha=0.7)
    plt.plot(range(len(train), len(ts)), test, label='Test', alpha=0.7)
    plt.plot(range(len(train), len(ts)), forecast, label='Forecast', linestyle='--', alpha=0.7)
    plt.axvline(x=train_size, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Holt-Winters Forecasting')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('holt_winters_forecast.png')
    print("Plot saved as 'holt_winters_forecast.png'\n")
    
    return fitted_model

def decompose_time_series():
    """Time Series Decomposition"""
    if not STATSMODELS_AVAILABLE:
        print("Decomposition requires statsmodels. Skipping...\n")
        return
    
    print("=" * 50)
    print("TIME SERIES DECOMPOSITION")
    print("=" * 50)
    
    ts = generate_time_series()
    
    # Decompose
    result = seasonal_decompose(ts, model='additive', period=12)
    
    # Plot
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    
    result.observed.plot(ax=axes[0], title='Original')
    axes[0].set_ylabel('Observed')
    
    result.trend.plot(ax=axes[1], title='Trend')
    axes[1].set_ylabel('Trend')
    
    result.seasonal.plot(ax=axes[2], title='Seasonal')
    axes[2].set_ylabel('Seasonal')
    
    result.resid.plot(ax=axes[3], title='Residual')
    axes[3].set_ylabel('Residual')
    
    plt.tight_layout()
    plt.savefig('time_series_decomposition.png')
    print("Decomposition plot saved as 'time_series_decomposition.png'\n")
    
    return result

if __name__ == "__main__":
    ts = moving_average_forecast()
    exponential_smoothing()
    
    if STATSMODELS_AVAILABLE:
        arima_model = demonstrate_arima()
        sarima_model = demonstrate_sarima()
        hw_model = demonstrate_hw_exponential_smoothing()
        decomposition = decompose_time_series()
    else:
        print("\nFor advanced time series analysis, install statsmodels:")
        print("pip install statsmodels")
