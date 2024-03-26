import pandas as pd
import numpy as np

data = pd.read_csv('E-commerce Customer Behavior - Sheet1.csv')

to_drop = ['Membership Type', 'Average Rating', 'Discount Applied', 'Days Since Last Purchase']

data_filtered = data.drop(to_drop, inplace=True, axis=1)

data_filtered.head()
