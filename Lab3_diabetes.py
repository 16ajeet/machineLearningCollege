"""
Part A: Minkowski Distance
	1.	Load the Diabetes dataset
	2.	Write a function minkowski_distance(x, y, p) that calculates the Minkowski distance of order p
	4.	Using the function, calculate the following distances
	•	L1 distance (p=1)
	•	L2 distance (p=2)
	•	L3 distance (p=3)
	5.	Print the results with appropriate labels.

⸻
Part B: Linear Regression
	1.	From the Diabetes dataset, use BMI (bmi) as the independent variable (X) and the disease progression (target) as the dependent variable (y).
	2.	Print the slope (coefficient) and intercept of the regression line.
	3.	Predict the disease progression score for a patient with a BMI value of 0.05.
	4.	Plot a scatter plot of BMI vs target along with the regression line.
"""

import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes

diabetes = load_diabetes()
X = diabetes.data   #fetches the data from diabetes and stores in x
y = diabetes.target

def minkowski_distance(x, y, p):
    if len(x) != len(y):
        raise ValueError("Vectors must be same length")
    total = 0
    for i in range(len(x)):
        total += abs(x[i] - y[i]) ** p
    return total ** (1/p)

# 3. Take first two patients
x1 = X[0]
x2 = X[1]

# calculating distance for p = 1, 2 and 3
d1 = minkowski_distance(x1, x2, 1)
d2 = minkowski_distance(x1, x2, 2)
d3 = minkowski_distance(x1, x2, 3)

# 5. Print results
print("Part A: Minkowski Distances")
print(f"L1 distance (p=1): {d1:.4f}")
print(f"L2 distance (p=2): {d2:.4f}")
print(f"L3 distance (p=3): {d3:.4f}")

# 1. Independent variable: BMI (column index 2)
X_bmi = [row[2] for row in X]  # extract BMI feature
y_target = list(y)             # convert numpy array to list

# 2. Compute slope and intercept using least squares
x_mean = sum(X_bmi) / len(X_bmi)
y_mean = sum(y_target) / len(y_target)

num = 0
den = 0
for i in range(len(X_bmi)):
    num += (X_bmi[i] - x_mean) * (y_target[i] - y_mean)
    den += (X_bmi[i] - x_mean) ** 2

slope = num / den
intercept = y_mean - slope * x_mean

print("\nPart B: Linear Regression")
print(f"Slope (coefficient): {slope:.4f}")
print(f"Intercept: {intercept:.4f}")

# 3. Predict target for BMI = 0.05
bmi_value = 0.05
prediction = slope * bmi_value + intercept
print(f"Predicted disease progression for BMI={bmi_value}: {prediction:.4f}")

# 4. Scatter plot with regression line
plt.scatter(X_bmi, y_target, color="blue", alpha=0.5, label="Data points")

# Regression line (manual prediction for each X)
y_pred_line = [slope * x + intercept for x in X_bmi]
plt.plot(X_bmi, y_pred_line, color="red", linewidth=2, label="Regression line")

plt.xlabel("BMI")
plt.ylabel("Disease Progression")
plt.title("Linear Regression: BMI vs Disease Progression")
plt.legend()
plt.show()

