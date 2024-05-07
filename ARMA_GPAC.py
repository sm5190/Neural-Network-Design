import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

import seaborn as sns
import pandas as pd

def calculate_gpac(ry, max_lag_j, max_lag_k):
    gpac_table = np.zeros((max_lag_j, max_lag_k))

    # Loop over the lags
    for j in range(max_lag_j):
        for k in range(max_lag_k):
            R_matrix = np.array([[ry[abs(i - m)] for m in range(k + 1)] for i in range(j, j + k + 1)])
            R_vector = np.array([ry[j + m + 1] for m in range(k + 1)])
            if np.linalg.matrix_rank(R_matrix) == R_matrix.shape[1]:

                phi = np.linalg.lstsq(R_matrix, R_vector, rcond=None)[0]
                gpac_table[j, k] = phi[-1]
                #gpac_table[j, k] = np.nan # Take the last element of the solution

    return gpac_table

ar_coeffs=[1, 0.5, 0.2]
ma_coeffs=[1, 0.5, -0.4]
arma_process = sm.tsa.ArmaProcess(ar_coeffs, ma_coeffs)

l=20000
ry= arma_process.acf(lags=l)
ry1=ry[::-1]
ry2=np.concatenate((np.reshape(ry1, l), ry[1:]))


def gpac_table_to_dataframe(gpac_table):
    df = pd.DataFrame(gpac_table)
    df.index.name = 'j'
    df.columns.name = 'k'
    return df

gpac_table = calculate_gpac(ry, max_lag_j=7, max_lag_k=7)

gpac_df = gpac_table_to_dataframe(gpac_table)
# Replace 0s with NaN
gpac_df = gpac_df.replace(0, np.nan)

#gpac_df.iloc[3:] = gpac_df.iloc[3:].replace(0, np.nan)
print(gpac_df.to_string())

new_x_labels = np.arange(1, len(gpac_df.columns) + 1)

sns.heatmap(gpac_df, annot=True, fmt='.2f', cmap='coolwarm', xticklabels=new_x_labels)

plt.title('GPAC Table')


# Show the plot
plt.show()