#!/usr/bin/env python
# coding: utf-8
#Demand Forcasting for predicting inventory demand using Random Forest Algorithm

print (' Demand forecasting to predict Inventory required in future')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.ensemble import RandomForestRegressor
from rfpimp import *
#get_ipython().run_line_magic('matplotlib', 'inline')

past_df = pd.read_excel('past_data.xlsx')
past_df.head(10)
np.any(np.isnan(past_df),axis = 0)

bymonth = past_df.groupby(['Item no','Sale year','Sale month'])
groupby_df = bymonth.sum()['Sales']
groupby_df.to_excel('monthly_Sales.xlsx')
print( "output file created: monthly_Sales.xlsx to analyse past data" )
itemlist = groupby_df.index.levels
itemlist = [list(x) for x in itemlist][0]
#itemscount = round(len(itemlist))

for X in itemlist:
    fig,axes = plt.subplots(nrows=1 , ncols = 1, figsize = (15,5))
    fig = groupby_df.loc[X].plot(legend =True, marker = 'o', markersize = 5 , markerfacecolor = 'red')
    fig.set_title(X)
    plt.xlabel('Time')
    plt.show()
    plt.tight_layout()

X = past_df[[ 'Item no', 'Sale year', 'Sale month','Discount', 'Price']]
y = past_df['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
rf = RandomForestRegressor(n_estimators= 1500, max_depth =7)
rf.fit(X_train,y_train)

#importance plot of individual features
imp = importances(rf, X_test, y_test) # permutation
viz = plot_importances(imp)
viz.view()
rf_predictions = rf.predict(X_test)
print ( 'Predictions score is {}'.format(rf.score(X_test,y_test)))
print( ' Root mean square error is {}'.format( np.sqrt(metrics.mean_squared_error(y_test, rf_predictions)))) 

#Show Scatter graph

plt.scatter(y_test,rf_predictions, color='blue')
plt.plot(y_test,rf_predictions,color='green', linewidth= 1, alpha = 0.5)
plt.xlabel('Actual sales')
plt.ylabel('Sales Predicted during Learning ')
plt.title(' Actual vs Predicted Sales learned through past data')
plt.show()

#Load future dataframe for predictions
future_df = pd.read_excel('future_data.xlsx')
print(future_df.head(10))

rf_future_predictions = rf.predict(future_df)
future_df['SALES_PREDICTED'] = rf_future_predictions

bymonthprediction = future_df.groupby(['Item no','Sale month'])
Gbymonthprediction = bymonthprediction.sum()['SALES_PREDICTED']
itemlist = Gbymonthprediction.index.levels
itemlist = [list(x) for x in itemlist][0]
itemscount = round(len(itemlist)/2)
for X in itemlist:
    fig,axes = plt.subplots(nrows=1 , ncols = 1, figsize = (15,5))
    fig = Gbymonthprediction.loc[X].plot(legend =True, marker = 'o', markersize = 5 , markerfacecolor = 'red')
    fig.set_title(X)
    plt.xlabel('Time')
    plt.show()

future_df.to_excel('ouput_predictions_Inventory.xlsx')
print( "\n" )
print( "output file created: ouput_predictions_Inventory.xlsx" )
exit()
