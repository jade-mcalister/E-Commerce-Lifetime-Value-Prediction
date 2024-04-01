from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# reads csv file
user_data = pd.read_csv(
    'C:\\Users\\veron\\OneDrive\\Documents\\School\\Machine Learning\\reg_data.csv')

#drops unnecessary columns
to_drop = ['customerid', "Unnamed: 0"]
user_data = user_data.drop(to_drop, axis=1)

# Replace 'no' with '0' and 'yes' with '1' in the 'churn' column
user_data['churn'] = user_data['churn'].replace({'No': '0', 'Yes': '1'})

dummy_data = pd.get_dummies(user_data)

# Calculate average cost per order
user_data['avg_charge'] = user_data['totalcharges']/user_data['monthlycharges']

# Calculate Purchase Frequency
purchase_frequency = (user_data[user_data.avg_charge > 1].shape[0]/user_data.shape[0])

# Calculate Churn Rate
churn_rate = user_data[user_data.churn == 1].sum()/len(user_data)

# Calculate Repeat Rate
#repeat_rate = 1 - churn_rate

# Calculate Profit Margin (est. 5% profit)
#user_data['profit_margin'] = user_data['total_spent']*0.05

# Calculate CLV
#user_data['customer_value'] = user_data['avg_order_value']*purchase_frequency/churn_rate


#user_data['CLV'] = user_data['customer_value']*user_data['profit_margin']
# define X features (everything except churn) and y targets (churn)
# y = dummy_data.churn.values
# X = dummy_data.drop('churn', axis = 1)
# saves dataframe column titles to list
#columns = X.columns

# instantiate a Min-Max scaling object
#minmax = MinMaxScaler()
# Fit and transform our feature data into a pandas dataframe
#X = pd.DataFrame(minmax.fit_transform(X))
# splits data
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1111)

print(churn_rate)