"""
predict student score based on number of hours they studied. use linear regression to predict score of a student
who studies for 9.5 hours
1. print the slope and intercept of the linear regression
2. plot the regression line with dataset
"""
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("C:/Users/Ajeet/Desktop/student_scores.csv")
hours = data["Hours"].tolist()      #converts panda series into python list and insert int hours
scores = data["Scores"].tolist()

#mean calculated by add hours and dividing by number of hours
x_mean = sum(hours) / len(hours)        #x_bar
y_mean = sum(scores) / len(scores)

num = 0
den = 0
for i in range(len(hours)):
    num += (hours[i] - x_mean) * (scores[i] - y_mean)       #calculates numerator -> (x_i-x_bar) * y_i-y_bar)
    den += (hours[i] - x_mean) ** 2     #calculates denominator

slope = num / den
intercept = y_mean - slope * x_mean

print("Linear Regression Results")
print(f"Slope (m): {slope:.4f}")
print(f"Intercept (b): {intercept:.4f}")

# 3. Prediction for 9.5 hours
study_hours = 9.5
predicted_score = slope * study_hours + intercept
print(f"Predicted score for {study_hours} hours: {predicted_score:.2f}")

# 4. Plot dataset and regression line
plt.scatter(hours, scores, color="blue", label="Data points")

y_pred_line = [slope * x + intercept for x in hours]
plt.plot(hours, y_pred_line, color="red", linewidth=2, label="Regression line")

plt.xlabel("Hours Studied")
plt.ylabel("Score")
plt.title("Student Score Prediction using Linear Regression")
plt.legend()
plt.show()

