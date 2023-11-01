import numpy as np
import pandas as pd
import scipy.stats as sp
import statsmodels.api as sm
import matplotlib.pyplot as plt

np.random.seed(1)  # Fix a seed for reproduction.

risk_simulation = np.zeros(21)
risk_limit = np.zeros(21)

n = 300
for j in range(20):
    p = 100 + 30 * j
    gamma = p / n

    beta = np.ones(p)
    x = np.random.normal(size=(n, p))
    eps = np.random.normal(size=(n, 1))
    y = x * beta + eps

    if n >= p:
        risk_limit[j] = gamma / (1 - gamma)

        bias = 0
        variance = np.linalg.inv(np.matmul(np.transpose(x), x))

        risk = bias + np.trace(variance)
        risk_simulation[j] = risk
    else:
        risk_limit[j] = np.dot(beta, beta) * (gamma - 1) / gamma + 1 / (gamma - 1)

        value, vector = np.linalg.eig(np.matmul(np.transpose(x), x))
        general_inv = np.zeros((p, p))


        for t in range(n):
            v = np.expand_dims(vector[p - 1 - t], axis=1)
            a = ((1 / value[p - 1 - t]) if (1 / value[p - 1 - t].real) ** 2 < 10000 else 0)
            general_inv = general_inv + (np.matmul(v, np.transpose(v)) * a)
        PI = np.zeros((p, p))
        for t in range(n, p):
            PI = PI + np.matmul(v, np.transpose(v))
        bias = np.matmul(np.expand_dims(beta, axis=1).T, np.matmul(PI, np.expand_dims(beta, axis=1) ))

        risk = bias[0] + np.trace(general_inv)
        risk_simulation[j] = risk[0]


print(risk_limit, risk_simulation)
plt.plot(risk_simulation, label='simulation')
plt.plot(risk_limit, label='limit')
plt.show(legend=True)