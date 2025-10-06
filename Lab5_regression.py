# Question 3: Multiple Regression (OLS)
# Part A
# Tasks:
# • Fit regression: Score = β0 + β1·Hours_Study + β2·Hours_Sleep.
# • Estimate coefficients (β0, β1, β2) and R².
# • Interpret coefficients.
# • Plot predicted vs actual scores.
# Part B
# • Import the Diabetes dataset from sklearn.datasets.
# • Use features bmi and bp to predict target.
# # • Fit regression model and report coefficients and R².

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes

Hours_Study = np.array([1,2,3,4,5,6,7,8])
Hours_sleep = np.array([8,7,7,6,6,5,5,4])
Scores = np.array([35.2,37.9,44.3,46.8,53.1,55.9,62.2,64.0])

X = np.column_stack((np.ones(len(Hours_Study)), Hours_Study, Hours_sleep))
y = Scores.reshape(-1, 1)

beta = np.linalg.inv(X.T @ X) @ (X.T @ y)

y_pred = X @ beta
ss_total = np.sum((y - np.mean(y))**2)
ss_residual = np.sum((y - y_pred)**2)   
r_squared = 1 - (ss_residual / ss_total)

print("=== Part A ===")
print("Estimated Coefficients (β0, β1, β2):", beta.ravel())
print("R²:", r_squared)

plt.scatter(y, y_pred, color='blue')
plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--')
plt.xlabel('Actual Scores')
plt.ylabel('Predicted Scores')
plt.title('Predicted vs Actual Scores')
plt.show()

