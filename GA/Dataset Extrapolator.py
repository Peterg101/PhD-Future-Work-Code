import pandas as pd
import numpy as np

initial_data = pd.read_csv('/Users/petergoon/Documents/Coding/Synthetic Data/separated_comp_vals.csv')
initial_data.columns = ['Material Code', 'Real', 'Imaginary']
print(initial_data)
interval = round(0.05, 2)

x = 8.2
y = 1.001

for i in range(0, 85):
    print(round(x+(i*interval),2))
    initial_data['Material Code'] = initial_data['Material Code'].astype(int)
    initial_data['Real']=initial_data['Real']*y
    initial_data['Imaginary']=initial_data['Imaginary']*y
    print(initial_data)
    initial_data.to_csv(f'/Users/petergoon/Documents/Coding/Synthetic Data/GA/Dataset 1/{round(x, 2)}_data.csv', index= False)
    x+= interval

x = 8.2
dataset_2 = pd.read_csv('/Users/petergoon/Documents/Coding/Synthetic Data/complex_separated_comp_vals.csv')
dataset_2.columns = ['Material Code', 'Real', 'Imaginary']

for i in range(0,85):
    print(round(x+(i*interval),2))
    dataset_2['Material Code'] = dataset_2['Material Code'].astype(int)
    dataset_2['Real']=dataset_2['Real']*y
    dataset_2['Imaginary']=dataset_2['Imaginary']*y
    print(dataset_2)
    dataset_2.to_csv(f'/Users/petergoon/Documents/Coding/Synthetic Data/GA/Dataset 2/{round(x, 2)}_data.csv', index= False)
    x+= interval

