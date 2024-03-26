import csv
import pandas as pd
import numpy as np
data = pd.read_csv(
   'C:\\Users\\veron\\PycharmProjects\\E-Commerce-Lifetime-Value-Prediction\\E-commerce Customer Behavior - Sheet1.csv')

to_drop = ['Membership Type', 'Average Rating', 'Discount Applied', 'Days Since Last Purchase']


print(data)
