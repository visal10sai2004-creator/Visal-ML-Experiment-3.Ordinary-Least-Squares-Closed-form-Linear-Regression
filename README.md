## Experiment-3.Ordinary Least Squares (Closed-form) Linear Regression

## AIM:

Implement OLS via normal equations and compare to scikit-learn.

## ALGORITHM:
 
Normal equation : 𝛽̂ = (𝑋𝑇𝑋)−1𝑋𝑇y

## THEORY:

OLS minimizes squared residuals; closed-form solution requires invertible Gram matrix.

## PROGRAM:

```

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

x = np.array([[50, 10], [80, 5], [60, 8], [100, 21], [45, 12]])
y = np.array([150, 240, 180, 320, 140])

x0 = np.hstack((np.ones((x.shape[0], 1)), x))

beta_hat = np.linalg.inv(x0.T.dot(x0)).dot(x0.T).dot(y)
y_pred_closed = x0.dot(beta_hat)

model = LinearRegression().fit(x, y)
y_pred_sk = model.predict(x)

print('Closed-form beta:', np.round(beta_hat, 3))
print('MSE (closed):', mean_squared_error(y, y_pred_closed))
print('MSE (sklearn):', mean_squared_error(y, y_pred_sk))

```

## OUTPUT:

```

Closed-form beta: [-15.669   3.116   1.15 ]
MSE (closed): 1.2067488866050045
MSE (sklearn): 1.2067488866049982


```

## RESULT:

Students see how closed-form OLS yields coefficients; numerical stability note: prefer SVD in practice.
