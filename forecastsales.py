#!/usr/bin/env python
# coding: utf-8
from shutil import copyfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from rfpimp import *
import pyodbc
import sqlalchemy 
import pymssql
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#%matplotlib inline
import plotly
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode
plotly.offline.init_notebook_mode(connected=True)


print (' Forcasting to predict Inventory Demand using Random Forest Algorithm')
print( "\n" )
print( "\n" )


conn_str = (
    r'DRIVER={SQL Server};'
    r'SERVER=IN2NPDVDCOM01;'
    r'UID=sa;'
    r'PWD=Ate@m2013;'
    r'DATABASE=EP20132X;'
    r'Trusted_Connection=no;'
)
learn_df = pd.read_sql("select * from PREDICTIONS", pyodbc.connect(conn_str))
pyodbc.Connection.close
learn_df.head(10)

# define a function for plotly layout
def plot_layout(ITEM):
    layout = go.Layout(
    title=go.layout.Title(
        text=int(ITEM),
         xref='paper',
         x=0
     ),
     xaxis=go.layout.XAxis(
         title=go.layout.xaxis.Title(
             text='Weeks',
             font=dict(
                 family='Comic sans, monospace',
                 size=18,
                 color='#7f7f7f'
             )
         )
     ),
     yaxis=go.layout.YAxis(
         title=go.layout.yaxis.Title(
             text='Count',
             font=dict(
                 family='Comic sans, monospace',
                 size=18,
                 color='#7f7f7f'
             )
         )
     )
     )
    return layout


# plot the past data yearwise and weekwise where each line represents year  
byitemyearweek = learn_df.groupby(['ITEMEDP','SALE_YEAR','SALE_WEEK'])
Gbyitemyearweek = byitemyearweek.sum()['INVENTORY_USED']
itemlist = Gbyitemyearweek.index.levels
itemlist = [list(x) for x in itemlist][0]
for item in itemlist:
    yearlist = learn_df[learn_df['ITEMEDP']== item]['SALE_YEAR'].unique()
    yearlist = list(yearlist)
    i = -1
    XX_INDEX = []
    XX_VALUE = []
    YEAR_=[]
    for year in yearlist:
        i = i + 1
        XX_INDEX.append(list(Gbyitemyearweek.loc[item].loc[year].index))
        XX_VALUE.append(list(Gbyitemyearweek.loc[item].loc[year].data)) 
    while (i > -1):
        YEAR_.append(go.Scatter( x = XX_INDEX[i], y = XX_VALUE[i], name = int(yearlist[i]), mode = 'lines+markers' ))
        i = i - 1
    layout = plot_layout(item)
    data = YEAR_
    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig , filename=str(int(item)))

# split in train test
X = learn_df[[ 'ITEMEDP', 'UNIT_PRICE', 'SALE_YEAR', 'SALE_WEEK','WAREHOUSE_LOCATION', 'DISCOUNT_PERCENTAGE']]
y = learn_df['INVENTORY_USED']
print( "\n" )
print("Using features  'ITEMEDP', 'UNIT_PRICE', 'SALE_YEAR', 'SALE_WEEK','WAREHOUSE_LOCATION', 'DISCOUNT_PERCENTAGE' to predict future Inventory required")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101) 

# train the model
rf = RandomForestRegressor(n_estimators= 1000, max_depth =9)
rf.fit(X_train,y_train)

print( "\n" )
print ( "Showing up the importance plot of individual features ")

imp = importances(rf, X_test, y_test) # permutation
viz = plot_importances(imp)
viz.view()

rf_predictions = rf.predict(X_test)

print("\n")
print ( 'Predictions score is {}'.format(rf.score(X_test,y_test)))

print("\n")
print ( "Showing up the Scatter graph ")

plt.scatter(y_test,rf_predictions, color='red')
plt.plot(y_test,rf_predictions,color='blue', linewidth= 0.5, alpha = 0.5)
plt.xlabel('Actual values')
plt.ylabel('Predicted during Learning ')
plt.title(' Actual vs Predicted on past data')
plt.show()


print("\n")
print( ' Root mean square error is {}'.format( np.sqrt(metrics.mean_squared_error(y_test, rf_predictions)))) 


print("\n")
print("\n")
print( 'Learning is done. Program is ready for predictions')


future_df = pd.read_excel('input_future.xlsx')
print("\n")
print("\n")
print("Input file uploaded for predictions, showing top 10 records")
print("\n")
print(future_df.head(10))

#predict the future sales
rf_predictions_future = rf.predict(future_df)


future_df['SALES_PREDICTED'] = rf_predictions_future.round(decimals= 0)
print( "\n" )
print( "\n" )
print( " Showing pedicted Sales graphically" )
print( "\n" )
byitemyearweek_future = future_df.groupby(['ITEMEDP','SALE_YEAR','SALE_WEEK'])
Gbyitemyearweek_future = byitemyearweek_future.sum()['SALES_PREDICTED']
itemlist_future = Gbyitemyearweek_future.index.levels
itemlist_future = [list(x) for x in itemlist_future][0]
byitemyearweek = learn_df.groupby(['ITEMEDP','SALE_YEAR','SALE_WEEK'])
Gbyitemyearweek = byitemyearweek.sum()['INVENTORY_USED']
# write a file showing past and future sales yearwise and weekwise
itemlist = Gbyitemyearweek.index.levels
itemlist = [list(x) for x in itemlist][0]
writer = pd.ExcelWriter('Weekly Sales yearwise.xlsx')
# plot sales per year per product per plot dynamically 
for item in itemlist_future:
    yearlist_future = future_df[future_df['ITEMEDP']== item]['SALE_YEAR'].unique()
    yearlist_future = list(yearlist_future)
    i = -1
    XX_INDEX_future = []
    XX_VALUE_future = []
    YEAR_=[]
    df_predicted_comparision = pd.DataFrame(index = np.arange(1,53), data = [], columns= [])
    for year in yearlist_future:
        i = i + 1
        XX_INDEX_future.append(list(Gbyitemyearweek_future.loc[item].loc[year].index))
        XX_VALUE_future.append(list(Gbyitemyearweek_future.loc[item].loc[year].data))
        df_predicted_comparision[year] = list(Gbyitemyearweek_future.loc[item].loc[year].data)
    itemfound = 0
    for item2 in itemlist:
        if item == item2:
            itemfound = 1
            yearlist = learn_df[learn_df['ITEMEDP']== item]['SALE_YEAR'].unique()
            yearlist = list(yearlist)
            j = -1
            XX_INDEX = []
            XX_VALUE = []
            for year2 in yearlist:
                j = j + 1
                XX_INDEX.append(list(Gbyitemyearweek.loc[item2].loc[year2].index))
                XX_VALUE.append(list(Gbyitemyearweek.loc[item2].loc[year2].data))
                df_predicted_comparision[year2] = list(Gbyitemyearweek.loc[item2].loc[year2].data)
                # plot the past data yearwise and weekwise where each line represents year. Future sales will be compared with past sales in a single plot per productid  
            while (j > -1):
                if i > -1:
                    YEAR_.append(go.Scatter( x = XX_INDEX_future[i], y = XX_VALUE_future[i], name = int(yearlist_future[i]),  mode = 'lines+markers'   ))
                YEAR_.append(go.Scatter( x = XX_INDEX[j], y = XX_VALUE[j], name =int(yearlist[i]),  mode = 'lines+markers'  ))
                j = j - 1
                i = i -1
            layout = plot_layout(item)
            data = YEAR_
            fig = go.Figure(data=data, layout=layout)
            plotly.offline.plot(fig , filename=str(int(item)))
    # for new product having no past records
    if itemfound == 0:
        while (i > -1):
            YEAR_.append(go.Scatter( x = XX_INDEX_future[i], y = XX_VALUE_future[i], name = int(yearlist_future[i]) , mode = 'lines+markers'   ))
            i = i - 1
        data = YEAR_
        plotly.offline.plot(data , filename=str(int(item)))
    df_predicted_comparision.to_excel(writer, sheet_name= str(item) )
writer.save()
writer.close()
future_df.to_excel('ouput_predictions_Inventory.xlsx')
print( "\n" )
print( "output file created: Weekly Sales yearwise.xlsx" )
print( "output file created: ouput_predictions_Inventory.xlsx" )
print( "Winding up the kernel. Good Bye!!" )
#exit()

