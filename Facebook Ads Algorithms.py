Python 3.12.3 (tags/v3.12.3:f6650f9, Apr  9 2024, 14:05:25) [MSC v.1938 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> import pandas as pd
>>> data = {
...     'Month': ['Apr-24', 'Mar-24', 'Feb-24', 'Jan-24', 'Dec-23', 'Nov-23', 'Oct-23', 'Sep-23', 'Aug-23', 'Jul-23', 'Jun-23', 'May-23'],
...     'Cost': [10483, 18818, 18499, 86363, 91806, 31185, 22323, 19758, 18335, 12373, 4712, 3919],
...     'Purchases': [40, 58, 33, 96, 194, 61, 20, 36, 34, 39, 18, 17],
...     'Revenue': [104319, 129358, 84193, 265885, 465640, 204880, 44720, 66739, 62734, 68115, 34842, 15123],
...     'Clicks': [4169, 5793, 2690, 20032, 22264, 8310, 5985, 5652, 7637, 7223, 2401, 1596],
...     'Cost per Click': [2.5, 3.2, 6.9, 4.3, 4.1, 3.8, 3.7, 3.5, 2.4, 1.7, 2.0, 2.5],
...     'Click to purchase conversion': [1.0, 1.0, 1.2, 0.5, 0.9, 0.7, 0.3, 0.6, 0.4, 0.5, 0.7, 1.1],
...     'CAC': [262, 324, 561, 900, 473, 511, 1116, 549, 539, 317, 262, 231],
...     'ROAS': [10.0, 6.9, 4.6, 3.1, 5.1, 6.6, 2.0, 3.4, 3.4, 5.5, 7.4, 3.9],
...     'AOV': [2607.976, 2230.308966, 2551.312121, 2769.635417, 2400.208454, 3358.683607, 2236.0175, 1853.852778, 1845.122941, 1746.525641, 1935.666667, 889.5941176]
... }
>>> df = pd.DataFrame(data)
>>> print(df)
     Month   Cost  Purchases  Revenue  Clicks  Cost per Click  Click to purchase conversion   CAC  ROAS          AOV
0   Apr-24  10483         40   104319    4169             2.5                           1.0   262  10.0  2607.976000
1   Mar-24  18818         58   129358    5793             3.2                           1.0   324   6.9  2230.308966
2   Feb-24  18499         33    84193    2690             6.9                           1.2   561   4.6  2551.312121
3   Jan-24  86363         96   265885   20032             4.3                           0.5   900   3.1  2769.635417
4   Dec-23  91806        194   465640   22264             4.1                           0.9   473   5.1  2400.208454
5   Nov-23  31185         61   204880    8310             3.8                           0.7   511   6.6  3358.683607
6   Oct-23  22323         20    44720    5985             3.7                           0.3  1116   2.0  2236.017500
7   Sep-23  19758         36    66739    5652             3.5                           0.6   549   3.4  1853.852778
8   Aug-23  18335         34    62734    7637             2.4                           0.4   539   3.4  1845.122941
9   Jul-23  12373         39    68115    7223             1.7                           0.5   317   5.5  1746.525641
10  Jun-23   4712         18    34842    2401             2.0                           0.7   262   7.4  1935.666667
11  May-23   3919         17    15123    1596             2.5                           1.1   231   3.9   889.594118
>>>
KeyboardInterrupt
>>> df['Month'] = pd.to_datetime(df['Month'], format='%b-%y')
>>> df['Month_num'] = df['Month'].dt.month
>>> df['Year'] = df['Month'].dt.year
>>> missing_values = df.isnull().sum()
>>> print("Missing values:\n", missing_values)
Missing values:
 Month                           0
Cost                            0
Purchases                       0
Revenue                         0
Clicks                          0
Cost per Click                  0
Click to purchase conversion    0
CAC                             0
ROAS                            0
AOV                             0
Month_num                       0
Year                            0
dtype: int64
>>> total_cost = df['Cost'].sum()
>>> total_purchases = df['Purchases'].sum()
>>> total_revenue = df['Revenue'].sum()
>>> avg_cpc = df['Cost per Click'].mean()
>>> click_to_purchase_conversion = df['Click to purchase conversion'].mean()
>>> cac = df['CAC'].mean()
>>> roas = df['ROAS'].mean()
>>> aov = df['AOV'].mean()
>>>
>>> print("Total Cost:", total_cost)
Total Cost: 338574
>>> print("Total Purchases:", total_purchases)
Total Purchases: 646
>>> print("Total Revenue:", total_revenue)
Total Revenue: 1546548
>>> print("Average Cost per Click (CPC):", avg_cpc)
Average Cost per Click (CPC): 3.3833333333333333
>>> print("Click to Purchase Conversion Rate:", click_to_purchase_conversion)
Click to Purchase Conversion Rate: 0.7416666666666667
>>> print("Customer Acquisition Cost (CAC):", cac)
Customer Acquisition Cost (CAC): 503.75
>>> print("Return on Ad Spend (ROAS):", roas)
Return on Ad Spend (ROAS): 5.158333333333333
>>> print("Average Order Value (AOV):", aov)
Average Order Value (AOV): 2202.0753508000003
>>> import matplotlib.pyplot as plt
>>> plt.figure(figsize=(12, 6))
<Figure size 1200x600 with 0 Axes>
>>> plt.plot(df['Month'], df['Cost'], label='Cost')
[<matplotlib.lines.Line2D object at 0x000001FAA0C513A0>]
>>> plt.plot(df['Month'], df['Purchases'], label='Purchases')
[<matplotlib.lines.Line2D object at 0x000001FAA0C85460>]
>>> plt.plot(df['Month'], df['Revenue'], label='Revenue')
[<matplotlib.lines.Line2D object at 0x000001FAA0C85B80>]
>>> plt.xlabel('Month')
Text(0.5, 0, 'Month')
>>> plt.ylabel('Amount')
Text(0, 0.5, 'Amount')
>>> plt.title('Trends Over Time')
Text(0.5, 1.0, 'Trends Over Time')
>>> plt.legend()
<matplotlib.legend.Legend object at 0x000001FAA0C531A0>
>>> plt.show()
>>> correlation_matrix = df.corr()
>>> import seaborn as sns
>>> plt.figure(figsize=(12, 8))
<Figure size 1200x800 with 0 Axes>
>>> sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
<Axes: >
>>> plt.title('Correlation Matrix')
Text(0.5, 1.0, 'Correlation Matrix')
>>> plt.show()
>>> import matplotlib.pyplot as plt
>>> plt.figure(figsize=(10, 6))
<Figure size 1000x600 with 0 Axes>
>>> plt.bar(df['Month'], df['Purchases'], color='skyblue')
<BarContainer object of 12 artists>
>>> plt.xlabel('Month')
Text(0.5, 0, 'Month')
>>> plt.ylabel('Purchases')
Text(0, 0.5, 'Purchases')
>>> plt.title('Purchases per Month')
Text(0.5, 1.0, 'Purchases per Month')
>>> plt.xticks(rotation=45)
(array([19478., 19539., 19601., 19662., 19723., 19783.]), [Text(19478.0, 0, '2023-05'), Text(19539.0, 0, '2023-07'), Text(19601.0, 0, '2023-09'), Text(19662.0, 0, '2023-11'), Text(19723.0, 0, '2024-01'), Text(19783.0, 0, '2024-03')])
>>> plt.show()
>>> plt.figure(figsize=(10, 6))
<Figure size 1000x600 with 0 Axes>
>>> plt.bar(df['Month'], df['Cost per Click'], color='salmon')
<BarContainer object of 12 artists>
>>> plt.xlabel('Month')
Text(0.5, 0, 'Month')
>>> plt.ylabel('Cost per Click')
Text(0, 0.5, 'Cost per Click')
>>> plt.title('Cost per Click per Month')
Text(0.5, 1.0, 'Cost per Click per Month')
>>> plt.xticks(rotation=45)
(array([19478., 19539., 19601., 19662., 19723., 19783.]), [Text(19478.0, 0, '2023-05'), Text(19539.0, 0, '2023-07'), Text(19601.0, 0, '2023-09'), Text(19662.0, 0, '2023-11'), Text(19723.0, 0, '2024-01'), Text(19783.0, 0, '2024-03')])
>>> plt.show()
>>> from sklearn.model_selection import train_test_split
>>> from sklearn.linear_model import LinearRegression
>>> from sklearn.ensemble import RandomForestRegressor
>>> from sklearn.metrics import r2_score, mean_squared_error
>>> X = df[['Cost', 'Clicks', 'Cost per Click', 'Click to purchase conversion', 'CAC', 'AOV']]
>>> y_purchases = df['Purchases']
>>> y_revenue = df['Revenue']
>>> X_train, X_test, y_purchases_train, y_purchases_test = train_test_split(X, y_purchases, test_size=0.2, random_state=42)
>>> X_train, X_test, y_revenue_train, y_revenue_test = train_test_split(X, y_revenue, test_size=0.2, random_state=42)
>>>
>>> lr_purchases = LinearRegression()
>>> lr_purchases.fit(X_train, y_purchases_train)
LinearRegression()
>>> rf_purchases = RandomForestRegressor(random_state=42)
>>> rf_purchases.fit(X_train, y_purchases_train)
RandomForestRegressor(random_state=42)
>>>
>>> lr_revenue = LinearRegression()
>>> lr_revenue.fit(X_train, y_revenue_train)
LinearRegression()
>>>
>>> rf_revenue = RandomForestRegressor(random_state=42)
>>> rf_revenue.fit(X_train, y_revenue_train)
RandomForestRegressor(random_state=42)
>>>
>>> lr_purchases_score = lr_purchases.score(X_test, y_purchases_test)
>>> rf_purchases_score = rf_purchases.score(X_test, y_purchases_test)
>>> lr_revenue_score = lr_revenue.score(X_test, y_revenue_test)
>>> rf_revenue_score = rf_revenue.score(X_test, y_revenue_test)
>>>
>>> lr_purchases_score, rf_purchases_score, lr_revenue_score, rf_revenue_score
(-2.667521723733223, -0.9043676025917924, -0.8334621648079885, 0.22817742088719495)
>>>


>>> from sklearn.model_selection import cross_val_score
>>> from sklearn.ensemble import GradientBoostingRegressor
>>>
>>> # Define the features (revenue) and target variable (cpc)
>>> revenue = [
...     [104319], [129358], [84193], [265885], [465640], [204880],
...     [44720], [66739], [62734], [68115], [34842], [15123]
... ]
>>> cpc = [3.2, 2.5, 3.2, 6.9, 4.3, 5.1, 6.6, 2.0, 3.4, 3.4, 5.5, 7.4]
>>>
>>> # Create the GradientBoostingRegressor
>>> gb_revenue_to_cpc = GradientBoostingRegressor(random_state=42)
>>>
>>> # Perform cross-validation
>>> scores = cross_val_score(gb_revenue_to_cpc, revenue, cpc, cv=5, scoring='neg_mean_squared_error')
>>>
>>> from sklearn.model_selection import cross_val_score
>>> from sklearn.ensemble import GradientBoostingRegressor
>>> revenue = [
...     [104319], [129358], [84193], [265885], [465640], [204880],
...     [44720], [66739], [62734], [68115], [34842], [15123]
... ]
>>> cpc = [3.2, 2.5, 3.2, 6.9, 4.3, 5.1, 6.6, 2.0, 3.4, 3.4, 5.5, 7.4]
>>> gb_revenue_to_cpc = GradientBoostingRegressor(random_state=42)
>>> scores = cross_val_score(gb_revenue_to_cpc, revenue, cpc, cv=5, scoring='neg_mean_squared_error')
>>> mean_mse = -scores.mean()
>>>
>>> mean_mse
2.907709519386535
>>>



>>> import pandas as pd
>>>
>>> # Facebook Ads data
>>> data = {
...     "Month": ["Apr-24", "Mar-24", "Feb-24", "Jan-24", "Dec-23", "Nov-23", "Oct-23", "Sep-23", "Aug-23", "Jul-23", "Jun-23", "May-23"],
...     "Cost": [10483, 18818, 18499, 86363, 91806, 31185, 22323, 19758, 18335, 12373, 4712, 3919],
...     "Purchases": [40, 58, 33, 96, 194, 61, 20, 36, 34, 39, 18, 17],
...     "Revenue": [104319, 129358, 84193, 265885, 465640, 204880, 44720, 66739, 62734, 68115, 34842, 15123],
...     "Clicks": [4169, 5793, 2690, 20032, 22264, 8310, 5985, 5652, 7637, 7223, 2401, 1596]
... }
>>>
>>> # Create a DataFrame
>>> df = pd.DataFrame(data)
>>>
>>> # Convert Month to datetime format
>>> df['Month'] = pd.to_datetime(df['Month'], format='%b-%y')
>>>
>>> # Set Month as index
>>> df.set_index('Month', inplace=True)
>>>
>>> # Display the DataFrame
>>> print(df)
             Cost  Purchases  Revenue  Clicks
Month
2024-04-01  10483         40   104319    4169
2024-03-01  18818         58   129358    5793
2024-02-01  18499         33    84193    2690
2024-01-01  86363         96   265885   20032
2023-12-01  91806        194   465640   22264
2023-11-01  31185         61   204880    8310
2023-10-01  22323         20    44720    5985
2023-09-01  19758         36    66739    5652
2023-08-01  18335         34    62734    7637
2023-07-01  12373         39    68115    7223
2023-06-01   4712         18    34842    2401
2023-05-01   3919         17    15123    1596
>>> import matplotlib.pyplot as plt
>>>
>>> # Plot the Revenue data
>>> plt.figure(figsize=(12, 6))
<Figure size 1200x600 with 0 Axes>
>>> plt.plot(df['Revenue'], marker='o')
[<matplotlib.lines.Line2D object at 0x000002024806B2F0>]
>>> plt.title('Facebook Ads Revenue Over Time')
Text(0.5, 1.0, 'Facebook Ads Revenue Over Time')
>>> plt.xlabel('Month')
Text(0.5, 0, 'Month')
>>> plt.ylabel('Revenue')
Text(0, 0.5, 'Revenue')
>>> plt.grid(True)
>>> plt.show()
>>> from statsmodels.tsa.stattools import adfuller
>>>
>>> # Perform Augmented Dickey-Fuller test
>>> result = adfuller(df['Revenue'])
>>>
>>> # Print test statistic and p-value
>>> print('ADF Statistic:', result[0])
ADF Statistic: -3.1977370635906115
>>> print('p-value:', result[1])
p-value: 0.02011604109608485
>>>


>>> from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
>>> import matplotlib.pyplot as plt
>>>
>>> # Create ACF plot
>>> plot_acf(df['Revenue'])
<Figure size 640x480 with 1 Axes>
>>> plt.show()
>>>
>>> # Create PACF plot
>>> plot_pacf(df['Revenue'])
<Figure size 640x480 with 1 Axes>
>>> plt.show()
>>> plot_acf(df['Revenue'])
<Figure size 640x480 with 1 Axes>
>>> plt.show()
>>> plot_pacf(df['Revenue'])
<Figure size 640x480 with 1 Axes>
>>> plt.show()


>>> import pandas as pd
>>> from statsmodels.tsa.arima.model import ARIMA
>>> from sklearn.metrics import mean_squared_error
>>> data = {
...     "Month": ["Apr-24", "Mar-24", "Feb-24", "Jan-24", "Dec-23", "Nov-23", "Oct-23", "Sep-23", "Aug-23", "Jul-23", "Jun-23", "May-23"],
...     "Revenue": [104319, 129358, 84193, 265885, 465640, 204880, 44720, 66739, 62734, 68115, 34842, 15123]
... }
>>> df = pd.DataFrame(data)
>>> df['Month'] = pd.to_datetime(df['Month'], format='%b-%y')
>>> df.set_index('Month', inplace=True)
>>>
>>> model = ARIMA(df['Revenue'], order=(1, 1, 1))

>>> model_fit = model.fit()
>>> forecast_steps = 3  # Example: forecast next 3 months
>>> forecast_index = pd.date_range(start=df.index[-1], periods=forecast_steps + 1, freq='MS')[1:]
>>> predictions = model_fit.forecast(steps=forecast_steps, index=forecast_index)
>>> mse = mean_squared_error(df['Revenue'], model_fit.fittedvalues)
>>> print('Mean Squared Error:', mse)
Mean Squared Error: 11745203190.186872
>>>




>>> import numpy as np
>>> import pandas as pd
>>> from sklearn.preprocessing import MinMaxScaler
>>> from sklearn.metrics import mean_squared_error
>>> from statsmodels.tsa.arima.model import ARIMA
>>> from keras.models import Sequential
>>> from keras.layers import LSTM, Dense
>>> import matplotlib.pyplot as plt
>>>
>>> # Facebook Ads data
>>> data = {
...     'Month': ['Apr-24', 'Mar-24', 'Feb-24', 'Jan-24', 'Dec-23', 'Nov-23', 'Oct-23', 'Sep-23', 'Aug-23', 'Jul-23', 'Jun-23', 'May-23'],
...     'Revenue': [104319, 129358, 84193, 265885, 465640, 204880, 44720, 66739, 62734, 68115, 34842, 15123]
... }
>>> df = pd.DataFrame(data)
>>>
>>> # Normalize the data
>>> scaler = MinMaxScaler(feature_range=(0, 1))
>>> df['Revenue'] = scaler.fit_transform(np.array(df['Revenue']).reshape(-1, 1))
>>>
>>> # Prepare the data for LSTM
>>> def create_dataset(dataset, time_steps=1):
...     X, y = [], []
...     for i in range(len(dataset) - time_steps):
...         X.append(dataset[i:(i + time_steps), 0])
...         y.append(dataset[i + time_steps, 0])
...     return np.array(X), np.array(y)
...
>>> time_steps = 3
>>> X, y = create_dataset(df[['Revenue']].values, time_steps)
>>>
>>> # Reshape input to be [samples, time steps, features]
>>> X = np.reshape(X, (X.shape[0], X.shape[1], 1))
>>>
>>> # Split the data into training and testing sets
>>> train_size = int(len(X) * 0.8)
>>> X_train, X_test = X[:train_size], X[train_size:]
>>> y_train, y_test = y[:train_size], y[train_size:]
>>>



>>> # Build the LSTM model
>>> model = Sequential()
>>> model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
C:\Program Files\Python312\Lib\site-packages\keras\src\layers\rnn\rnn.py:204: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(**kwargs)
>>> model.add(LSTM(units=50))
>>> model.add(Dense(units=1))
>>> model.compile(optimizer='adam', loss='mean_squared_error')
>>>
>>> # Train the model
>>> model.fit(X_train, y_train, epochs=100, batch_size=32)
Epoch 1/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m2s←[0m 2s/step - loss: 0.2273
Epoch 2/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 18ms/step - loss: 0.2188
Epoch 3/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 52ms/step - loss: 0.2107
Epoch 4/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 19ms/step - loss: 0.2030
Epoch 5/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 19ms/step - loss: 0.1956
Epoch 6/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 19ms/step - loss: 0.1884
Epoch 7/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 19ms/step - loss: 0.1815
Epoch 8/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 19ms/step - loss: 0.1748
Epoch 9/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 19ms/step - loss: 0.1683
Epoch 10/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 19ms/step - loss: 0.1621
Epoch 11/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 19ms/step - loss: 0.1560
Epoch 12/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 17ms/step - loss: 0.1503
Epoch 13/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 19ms/step - loss: 0.1450
Epoch 14/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 19ms/step - loss: 0.1400
Epoch 15/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 21ms/step - loss: 0.1354
Epoch 16/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 18ms/step - loss: 0.1314
Epoch 17/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 18ms/step - loss: 0.1279
Epoch 18/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 19ms/step - loss: 0.1251
Epoch 19/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 19ms/step - loss: 0.1231
Epoch 20/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 18ms/step - loss: 0.1217
Epoch 21/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 19ms/step - loss: 0.1209
Epoch 22/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 19ms/step - loss: 0.1207
Epoch 23/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 19ms/step - loss: 0.1208
Epoch 24/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 18ms/step - loss: 0.1210
Epoch 25/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 19ms/step - loss: 0.1212
Epoch 26/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 18ms/step - loss: 0.1211
Epoch 27/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 19ms/step - loss: 0.1208
Epoch 28/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 21ms/step - loss: 0.1202
Epoch 29/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 19ms/step - loss: 0.1193
Epoch 30/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 19ms/step - loss: 0.1183
Epoch 31/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 17ms/step - loss: 0.1172
Epoch 32/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 19ms/step - loss: 0.1160
Epoch 33/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 19ms/step - loss: 0.1150
Epoch 34/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 20ms/step - loss: 0.1140
Epoch 35/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 19ms/step - loss: 0.1132
Epoch 36/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 19ms/step - loss: 0.1125
Epoch 37/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 19ms/step - loss: 0.1119
Epoch 38/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 18ms/step - loss: 0.1114
Epoch 39/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 19ms/step - loss: 0.1110
Epoch 40/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 19ms/step - loss: 0.1107
Epoch 41/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 19ms/step - loss: 0.1103
Epoch 42/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 19ms/step - loss: 0.1100
Epoch 43/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 19ms/step - loss: 0.1097
Epoch 44/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 19ms/step - loss: 0.1094
Epoch 45/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 20ms/step - loss: 0.1091
Epoch 46/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 19ms/step - loss: 0.1087
Epoch 47/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 18ms/step - loss: 0.1083
Epoch 48/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 19ms/step - loss: 0.1079
Epoch 49/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 19ms/step - loss: 0.1075
Epoch 50/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 19ms/step - loss: 0.1071
Epoch 51/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 19ms/step - loss: 0.1067
Epoch 52/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 20ms/step - loss: 0.1063
Epoch 53/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 19ms/step - loss: 0.1059
Epoch 54/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 18ms/step - loss: 0.1055
Epoch 55/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 18ms/step - loss: 0.1051
Epoch 56/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 18ms/step - loss: 0.1048
Epoch 57/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 19ms/step - loss: 0.1044
Epoch 58/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 19ms/step - loss: 0.1041
Epoch 59/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 20ms/step - loss: 0.1038
Epoch 60/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 20ms/step - loss: 0.1035
Epoch 61/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 19ms/step - loss: 0.1031
Epoch 62/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 19ms/step - loss: 0.1028
Epoch 63/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 21ms/step - loss: 0.1024
Epoch 64/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 19ms/step - loss: 0.1021
Epoch 65/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 19ms/step - loss: 0.1017
Epoch 66/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 20ms/step - loss: 0.1013
Epoch 67/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 20ms/step - loss: 0.1009
Epoch 68/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 20ms/step - loss: 0.1005
Epoch 69/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 18ms/step - loss: 0.1001
Epoch 70/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 19ms/step - loss: 0.0997
Epoch 71/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 19ms/step - loss: 0.0993
Epoch 72/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 19ms/step - loss: 0.0989
Epoch 73/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 19ms/step - loss: 0.0984
Epoch 74/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 19ms/step - loss: 0.0980
Epoch 75/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 19ms/step - loss: 0.0976
Epoch 76/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 19ms/step - loss: 0.0971
Epoch 77/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 19ms/step - loss: 0.0967
Epoch 78/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 19ms/step - loss: 0.0962
Epoch 79/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 20ms/step - loss: 0.0957
Epoch 80/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 19ms/step - loss: 0.0952
Epoch 81/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 19ms/step - loss: 0.0947
Epoch 82/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 19ms/step - loss: 0.0942
Epoch 83/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 20ms/step - loss: 0.0937
Epoch 84/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 19ms/step - loss: 0.0932
Epoch 85/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 19ms/step - loss: 0.0926
Epoch 86/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 17ms/step - loss: 0.0921
Epoch 87/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 19ms/step - loss: 0.0916
Epoch 88/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 18ms/step - loss: 0.0911
Epoch 89/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 19ms/step - loss: 0.0905
Epoch 90/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 19ms/step - loss: 0.0900
Epoch 91/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 19ms/step - loss: 0.0895
Epoch 92/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 19ms/step - loss: 0.0890
Epoch 93/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 18ms/step - loss: 0.0886
Epoch 94/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 19ms/step - loss: 0.0881
Epoch 95/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 19ms/step - loss: 0.0877
Epoch 96/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 20ms/step - loss: 0.0873
Epoch 97/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 19ms/step - loss: 0.0869
Epoch 98/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 19ms/step - loss: 0.0865
Epoch 99/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 18ms/step - loss: 0.0862
Epoch 100/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 18ms/step - loss: 0.0859
<keras.src.callbacks.history.History object at 0x0000020263D37230>
>>>
>>> # Make predictions
>>> train_predict = model.predict(X_train)
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 156ms/step
>>> test_predict = model.predict(X_test)
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 159ms/step
>>>
>>> # Invert predictions
>>> train_predict = scaler.inverse_transform(train_predict)
>>> y_train = scaler.inverse_transform([y_train])
>>> test_predict = scaler.inverse_transform(test_predict)
>>> y_test = scaler.inverse_transform([y_test])
>>>
>>> # Calculate root mean squared error
>>> train_score = np.sqrt(mean_squared_error(y_train[0], train_predict[:,0]))
>>> print('Train Score: %.2f RMSE' % train_score)
Train Score: 131878.54 RMSE
>>> test_score = np.sqrt(mean_squared_error(y_test[0], test_predict[:,0]))
>>> print('Test Score: %.2f RMSE' % test_score)
Test Score: 199870.96 RMSE
>>>


>>> # Plot the results
>>> plt.plot(df['Revenue'].values)
[<matplotlib.lines.Line2D object at 0x0000020263FFBF20>]
>>> plt.plot(np.concatenate((train_predict, test_predict)))
[<matplotlib.lines.Line2D object at 0x0000020265282EA0>]
>>> plt.show()
>>>