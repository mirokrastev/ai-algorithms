import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('Salary_Data.csv')
x = df[['YearsExperience']]
y = df['Salary']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

regressor = LinearRegression()

regressor.fit(x_train, y_train)  # Train the model with x_train and y_train
prediction = regressor.predict(x_test)  # Predict y_test for x_test

plt.scatter(x_test, y_test, color='red')  # Plot the real values
plt.scatter(x_test, prediction, color='blue')  # Plot the predicted values as blue color
# plt.plot(X_test, prediction)  # Draw linear line
plt.show()
