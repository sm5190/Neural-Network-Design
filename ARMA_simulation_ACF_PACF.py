import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def simulate_arma(n_samples, noise_variance, ar_coeffs, ma_coeffs):
    np.random.seed(5806)
    ar_params = np.array([1] + [-coeff for coeff in ar_coeffs])
    ma_params = np.array([1] + ma_coeffs)
    arma_process = sm.tsa.ArmaProcess(ar=ar_params, ma=ma_params)
    y = arma_process.generate_sample(nsample=n_samples, scale=np.sqrt(noise_variance))
    return y

# User inputs
n_samples = int(input("Enter the number of data samples: "))
noise_variance = float(input("Enter the variance of the white noise: "))

# AR coefficients
ar_order = int(input("Enter AR order: "))
ar_coeffs = []
for i in range(ar_order):
    coeff = float(input(f"Enter coefficient for AR lag {i+1}: "))
    ar_coeffs.append(coeff)

# MA coefficients
ma_order = int(input("Enter MA order: "))
ma_coeffs = []
for i in range(ma_order):
    coeff = float(input(f"Enter coefficient for MA lag {i+1}: "))
    ma_coeffs.append(coeff)

# Simulate the ARMA process
y = simulate_arma(n_samples, noise_variance, ar_coeffs, ma_coeffs)

# Plot the simulated process
plt.figure(figsize=(10, 4))
plt.plot(y)
plt.title(f'Simulated ARMA({ar_order},{ma_order}) Process')
plt.xlabel('Sample Number')
plt.ylabel('Value')
plt.grid(True)  # Optional: Add grid for better readability
plt.show()


def ACF_PACF_Plot(y, lags):
    acf = sm.tsa.stattools.acf(y, nlags=lags)
    pacf = sm.tsa.stattools.pacf(y, nlags=lags)
    fig = plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.title('ACF of the raw data')
    plot_acf(y, ax=plt.gca(), lags=lags)
    plt.subplot(122)
    plt.title('PACF of the raw data')
    plot_pacf(y, ax=plt.gca(), lags=lags)
    fig.tight_layout(pad=3)
    plt.show()


ACF_PACF_Plot(y, lags=20)