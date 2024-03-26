from sklearn.model_selection import train_test_split
import pandas as pd

# reads csv file
user_data = pd.read_csv('C:\\Users\\veron\\PycharmProjects\\E-Commerce-Lifetime-Value-Prediction\\E-commerce Customer Behavior - Sheet1.csv')

# drops unnecessary columns
to_drop = ['Membership Type', 'Average Rating', 'Discount Applied', 'Satisfaction Level']
user_data = user_data.drop(to_drop, axis=1)
print(user_data)

# defining X features and y targets
X = user_data.drop('Total Spend', axis=1)  # Features
y = user_data['Total Spend']  # Target

# splits data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1111)

user_data.head()