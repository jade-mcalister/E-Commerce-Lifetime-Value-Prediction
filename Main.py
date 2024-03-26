import csv
import pandas as pd
import numpy as np

#reads csv file
data = pd.read_csv(
   'C:\\Users\\veron\\PycharmProjects\\E-Commerce-Lifetime-Value-Prediction\\E-commerce Customer Behavior - Sheet1.csv')

#drops unnecessary columns
to_drop = ['Membership Type', 'Average Rating', 'Discount Applied', 'Satisfaction Level', 'Days Since Last Purchase']
data = data.drop(to_drop, axis=1)
print(data)
