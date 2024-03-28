from sklearn.model_selection import train_test_split
import pandas as pd

# reads csv file
user_data = pd.read_csv(
    'C:\\Users\\veron\\PycharmProjects\\E-Commerce-Lifetime-Value-Prediction\\E-commerce Customer Behavior - Sheet1.csv')

# drops unnecessary columns
to_drop = ['Membership Type', 'City', 'Average Rating', 'Discount Applied', 'Satisfaction Level']
user_data = user_data.drop(to_drop, axis=1)

# Replace 'Female' with '0' and 'Male' with '1' in the 'Gender' column
user_data['Gender'] = user_data['Gender'].replace({'Female': '0', 'Male': '1'})

#Change column names
# Change the name of columns
user_data.columns=['cust_id', 'gender', 'age', 'total_spent', 'items_purchased', 'days_since_last_purchase']

# Calculate average cost per order
user_data['avg_order_value'] = user_data['total_spent']/user_data['items_purchased']

#Calculate Purchase Frequency
purchase_frequency = (user_data[user_data.items_purchased > 1].shape[0]/user_data.shape[0])
# Repeat Rate

#Calculate Repeat Rate (looks like our repeat rate is 100% in this dataset)
repeat_rate=user_data[user_data.items_purchased > 1].shape[0]/user_data.shape[0]

#Calculate Churn Rate (this would be zero, so I added a few decimals to make this number divisible)
churn_rate = 1.0001 - repeat_rate

#Calculate Profit Margin (est. 5% profit)
user_data['profit_margin'] = user_data['total_spent']*0.05

#Calculate CLV
user_data['customer_value'] = user_data['avg_order_value']*purchase_frequency/churn_rate


user_data['CLV'] = user_data['customer_value']*user_data['profit_margin']
# defining X features and y targets
X = user_data.drop('CLV', axis=1)  # Features
y = user_data['CLV']  # Target

# splits data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1111)

print(user_data)

