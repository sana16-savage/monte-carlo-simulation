# monte-carlo-simulation
The project uses 1000 paths with the common assumption of stocks following a log normal distribution. It was tested on Palantir stock adding the data from past 5 yrs till 20 jan 2026. If keeping the assumptions constant expecting no market fluctuation its should predict prices accurately by  116.15 next year.
the following is my code - 

import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

ticker = 'PLTR'
data = yf.download(ticker, start='2020-01-01', end='2025-01-20')['Close']
# if data is a single-column DataFrame, convert to Series
if isinstance(data, pd.DataFrame):
    data = data.iloc[:, 0]

log_returns = np.log(1 + data.pct_change().dropna())
# Ensure these are plain floats (not pandas Series) for numpy arithmetic
mu = log_returns.mean()
if hasattr(mu, 'iloc'):
    mu = float(mu.iloc[0])
else:
    mu = float(mu)
var = log_returns.var()
if hasattr(var, 'iloc'):
    var = float(var.iloc[0])
else:
    var = float(var)
drift = mu - (0.5 * var)
sigma = log_returns.std()
if hasattr(sigma, 'iloc'):
    sigma = float(sigma.iloc[0])
else:
    sigma = float(sigma)

t_intervals = 252
interations = 1000
daily_returns = np.exp(drift + sigma * norm.ppf(np.random.rand(t_intervals, interations)))
price_paths = np.zeros_like(daily_returns)
# use the last close as scalar starting price
S0 = float(data.iloc[-1])
price_paths[0] = S0
for t in range(1, t_intervals):
    price_paths[t] = price_paths[t - 1] * daily_returns[t]

plt.figure(figsize=(10, 6))
plt.plot(price_paths)
plt.title(f'Simulated Stock Price Paths using Monte Carlo for {ticker}')
plt.xlabel('Days')
plt.ylabel('Price')
plt.show()
print(f"Expected price after {t_intervals} days: {price_paths[-1, :].mean():.2f}")


<img width="859" height="547" alt="image" src="https://github.com/user-attachments/assets/e385c3bb-d69c-473d-90d6-ba0dce4a0787" />
