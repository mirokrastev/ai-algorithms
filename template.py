"""
Template for preprocessing the data
"""
import pandas as pd
from sklearn.model_selection import train_test_split

"""
x: independent variable(s)
y: dependent variable
"""
df = pd.read_csv('Some_CSV.csv')
x = df[['...', '...']]
y = df['...']

# Uncomment if you have categorical column(s)/variable(s)
# x = pd.get_dummies(x, drop_first=True)

# Split the dataset to train and test. Adjust test_size if necessary
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
