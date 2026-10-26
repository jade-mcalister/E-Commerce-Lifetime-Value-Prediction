#LinearRegression v1
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# reads csv file
data = pd.read_csv('C:\\Users\\veron\\OneDrive\\Documents\\School\\Machine Learning\\reg_data.csv')

# Replace 'no' with '0' and 'yes' with '1'
data['churn'] = data['churn'].replace({'No': '0', 'Yes': '1'})
data['churn'] = pd.to_numeric(data['churn'])

# Replace 'female' with '0' and 'male' with '1'
data['gender'] = data['gender'].replace({'Female': '0', 'Male': '1'})
data['gender'] = pd.to_numeric(data['gender'])

# Replace 'no' with '0' and 'yes' with '1'
data['partner'] = data['partner'].replace({'No': '0', 'Yes': '1'})
data['partner'] = pd.to_numeric(data['partner'])

# Replace 'no' with '0' and 'yes' with '1'
data['dependents'] = data['dependents'].replace({'No': '0', 'Yes': '1'})
data['dependents'] = pd.to_numeric(data['dependents'])

# Replace 'no' with '0' and 'yes' with '1'
data['phoneservice'] = data['phoneservice'].replace({'No': '0', 'Yes': '1'})
data['phoneservice'] = pd.to_numeric(data['phoneservice'])

# Replace 'no' with '0' and 'yes' with '1'
data['multiplelines'] = data['multiplelines'].replace({'No': '0', 'No phone service': '0', 'Yes': '1'})
data['multiplelines'] = pd.to_numeric(data['multiplelines'])

# Replace 'no' with '0' and 'DSL' with '1' and 'Fiber Optic' with 2
data['internetservice'] = data['internetservice'].replace({'No': '0', 'DSL': '1', 'Fiber optic': '2'})
data['internetservice'] = pd.to_numeric(data['internetservice'])

# Replace 'no' with '0' and 'yes' with '1'
data['onlinesecurity'] = data['onlinesecurity'].replace({'No': '0', 'No internet service': '0', 'Yes': '1'})
data['onlinesecurity'] = pd.to_numeric(data['onlinesecurity'])

# Replace 'no' with '0' and 'yes' with '1'
data['onlinebackup'] = data['onlinebackup'].replace({'No': '0', 'No internet service': '0', 'Yes': '1'})
data['onlinebackup'] = pd.to_numeric(data['onlinebackup'])

# Replace 'no' with '0' and 'yes' with '1'
data['deviceprotection'] = data['deviceprotection'].replace({'No': '0', 'No internet service': '0', 'Yes': '1'})
data['deviceprotection'] = pd.to_numeric(data['deviceprotection'])

# Replace 'no' with '0' and 'yes' with '1'
data['techsupport'] = data['techsupport'].replace({'No': '0', 'No internet service': '0', 'Yes': '1'})
data['techsupport'] = pd.to_numeric(data['techsupport'])

# Replace 'no' with '0' and 'yes' with '1'
data['streamingtv'] = data['streamingtv'].replace({'No': '0', 'No internet service': '0', 'Yes': '1'})
data['streamingtv'] = pd.to_numeric(data['streamingtv'])

# Replace 'no' with '0' and 'yes' with '1'
data['streamingmovies'] = data['streamingmovies'].replace({'No': '0', 'No internet service': '0', 'Yes': '1'})
data['streamingmovies'] = pd.to_numeric(data['streamingmovies'])

# Replace 'month-to-month' with '0', 'one year' with 1 and 'two year' with '2'
data['contract'] = data['contract'].replace({'Month-to-month': '0', 'One year': '1', 'Two year': '2'})
data['contract'] = pd.to_numeric(data['contract'])

# Replace 'no' with '0' and 'yes' with '1'
data['paperlessbilling'] = data['paperlessbilling'].replace({'No': '0', 'No internet service': '0', 'Yes': '1'})
data['paperlessbilling'] = pd.to_numeric(data['paperlessbilling'])


# Calculate average cost per order
data['avg_charge'] = data['totalcharges']/data['monthlycharges']

# Calculate Purchase Frequency
purchase_frequency = (data[data.avg_charge > 1].shape[0]/data.shape[0])

# Calculate Churn Rate
churn_rate = data['churn'].mean()

# Calculate Repeat Rate
repeat_rate = 1 - churn_rate


# Calculate CLV
data['CLV'] = data['avg_charge']*purchase_frequency/churn_rate

# Drop unnecessary columns (if any)
data = data.drop(['customerid', 'paymentmethod'], axis=1)

# Convert categorical variables to numerical (if necessary)
# For example, you can use one-hot encoding or label encoding

# Split the data into features (X) and target variable (CLV)
X = data.drop('CLV', axis=1)
y = data['CLV']

# Split the data into training and testing sets
# You can adjust the test_size and random_state parameters as needed
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Add a column of ones to X_train for the bias term
X_train = np.c_[np.ones((X_train.shape[0], 1)), X_train]

# Implement linear regression manually using the normal equation
# theta = (X^T * X)^-1 * X^T * y
theta = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)

# Add a column of ones to X_test for the bias term
X_test = np.c_[np.ones((X_test.shape[0], 1)), X_test]

# Make predictions on the testing set
y_pred = X_test.dot(theta)

# Evaluate the model (e.g., calculate Mean Squared Error)
mse = np.mean((y_pred - y_test) ** 2)
print(f'Mean Squared Error: {mse}')

print(y_pred)
