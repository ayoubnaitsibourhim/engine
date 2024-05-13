Python 3.12.3 (tags/v3.12.3:f6650f9, Apr  9 2024, 14:05:25) [MSC v.1938 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> import pandas as pd
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> from sklearn.model_selection import train_test_split
>>> from sklearn.linear_model import LinearRegression
>>> from sklearn.ensemble import RandomForestRegressor
>>> from sklearn.metrics import mean_squared_error
>>> from statsmodels.tsa.arima.model import ARIMA
>>> from tensorflow.keras.models import Sequential
2024-05-12 11:33:17.883493: I tensorflow/core/util/port.cc:113] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2024-05-12 11:33:18.879607: I tensorflow/core/util/port.cc:113] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
>>> from tensorflow.keras.layers import LSTM, Dense
>>> from sklearn.preprocessing import MinMaxScaler
>>>
>>> # Load the data
>>> file_path = "C:\\Users\\t14\\Documents\\Facebook Ads Data.xlsx"
>>> df = pd.read_excel(file_path)
>>>
>>> # Define features and target variables
>>> X = df[['Cost', 'Clicks', 'Click to purchase conversion', 'ROAS', 'AOV']]
>>> y_ad_spend = df['Cost']
>>> y_conversion_rate = df['Click to purchase conversion']
>>> y_cpc = df['Cost per Click']
>>>
>>> # Split the data into training and test sets
>>> X_train, X_test, y_train, y_test = train_test_split(X, y_ad_spend, test_size=0.2, random_state=42)
>>>
>>> # Linear Regression for ad spend prediction
>>> lr_ad_spend = LinearRegression()
>>> lr_ad_spend.fit(X_train, y_train)
LinearRegression()
>>> y_pred_lr_ad_spend = lr_ad_spend.predict(X_test)
>>>
>>> # Random Forest Regression for ad spend prediction
>>> rf_ad_spend = RandomForestRegressor()
>>> rf_ad_spend.fit(X_train, y_train)
RandomForestRegressor()
>>> y_pred_rf_ad_spend = rf_ad_spend.predict(X_test)
>>>
>>> # Calculate RMSE for ad spend prediction
>>> rmse_lr_ad_spend = np.sqrt(mean_squared_error(y_test, y_pred_lr_ad_spend))
>>> rmse_rf_ad_spend = np.sqrt(mean_squared_error(y_test, y_pred_rf_ad_spend))
>>>
>>> # Time Series Analysis using ARIMA for ad spend
>>> arima_ad_spend = ARIMA(df['Cost'], order=(5,1,0))
>>> arima_ad_spend_fit = arima_ad_spend.fit()
>>> arima_ad_spend_pred = arima_ad_spend_fit.predict(start=len(df), end=len(df)+5, typ='levels')

>>>
>>> # LSTM for Time Series Analysis for ad spend
>>> scaler = MinMaxScaler()
>>> X_lstm = scaler.fit_transform(np.array(df['Cost']).reshape(-1,1))
>>> X_lstm_train, X_lstm_test, y_lstm_train, y_lstm_test = train_test_split(X_lstm, y_ad_spend, test_size=0.2, random_state=42)
>>> X_lstm_train = X_lstm_train.reshape(X_lstm_train.shape[0], X_lstm_train.shape[1], 1)
>>> X_lstm_test = X_lstm_test.reshape(X_lstm_test.shape[0], X_lstm_test.shape[1], 1)
>>>
>>> model_lstm = Sequential()
>>> model_lstm.add(LSTM(units=50, return_sequences=True, input_shape=(X_lstm_train.shape[1],1)))

>>> model_lstm.add(LSTM(units=50, return_sequences=False))
>>> model_lstm.add(Dense(units=1))
>>> model_lstm.compile(optimizer='adam', loss='mean_squared_error')
>>>
>>> model_lstm.fit(X_lstm_train, y_lstm_train, epochs=100, batch_size=32, verbose=1)
Epoch 1/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m2s←[0m 2s/step - loss: 2088441344.0000
Epoch 2/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 27ms/step - loss: 2088440960.0000
Epoch 3/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 32ms/step - loss: 2088440704.0000
Epoch 4/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 24ms/step - loss: 2088440064.0000
Epoch 5/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 26ms/step - loss: 2088439808.0000
Epoch 6/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 23ms/step - loss: 2088439552.0000
Epoch 7/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 29ms/step - loss: 2088439168.0000
Epoch 8/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 24ms/step - loss: 2088438912.0000
Epoch 9/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 23ms/step - loss: 2088438272.0000
Epoch 10/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 28ms/step - loss: 2088438016.0000
Epoch 11/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 22ms/step - loss: 2088437760.0000
Epoch 12/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 24ms/step - loss: 2088437248.0000
Epoch 13/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 23ms/step - loss: 2088436864.0000
Epoch 14/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 24ms/step - loss: 2088436352.0000
Epoch 15/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 24ms/step - loss: 2088435968.0000
Epoch 16/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 26ms/step - loss: 2088435712.0000
Epoch 17/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 27ms/step - loss: 2088435072.0000
Epoch 18/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 23ms/step - loss: 2088434304.0000
Epoch 19/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 27ms/step - loss: 2088434176.0000
Epoch 20/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 24ms/step - loss: 2088433408.0000
Epoch 21/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 24ms/step - loss: 2088433152.0000
Epoch 22/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 23ms/step - loss: 2088432512.0000
Epoch 23/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 24ms/step - loss: 2088431872.0000
Epoch 24/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 24ms/step - loss: 2088431104.0000
Epoch 25/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 24ms/step - loss: 2088430976.0000
Epoch 26/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 24ms/step - loss: 2088430208.0000
Epoch 27/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 23ms/step - loss: 2088429312.0000
Epoch 28/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 26ms/step - loss: 2088428928.0000
Epoch 29/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 27ms/step - loss: 2088428416.0000
Epoch 30/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 23ms/step - loss: 2088427520.0000
Epoch 31/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 23ms/step - loss: 2088426880.0000
Epoch 32/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 24ms/step - loss: 2088426112.0000
Epoch 33/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 23ms/step - loss: 2088425216.0000
Epoch 34/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 26ms/step - loss: 2088424320.0000
Epoch 35/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 23ms/step - loss: 2088423168.0000
Epoch 36/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 23ms/step - loss: 2088422784.0000
Epoch 37/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 23ms/step - loss: 2088421632.0000
Epoch 38/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 23ms/step - loss: 2088420864.0000
Epoch 39/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 28ms/step - loss: 2088419840.0000
Epoch 40/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 27ms/step - loss: 2088418816.0000
Epoch 41/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 23ms/step - loss: 2088417536.0000
Epoch 42/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 27ms/step - loss: 2088416640.0000
Epoch 43/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 24ms/step - loss: 2088415232.0000
Epoch 44/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 23ms/step - loss: 2088413824.0000
Epoch 45/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 23ms/step - loss: 2088412672.0000
Epoch 46/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 24ms/step - loss: 2088411392.0000
Epoch 47/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 24ms/step - loss: 2088409984.0000
Epoch 48/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 24ms/step - loss: 2088408448.0000
Epoch 49/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 23ms/step - loss: 2088407040.0000
Epoch 50/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 24ms/step - loss: 2088405504.0000
Epoch 51/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 24ms/step - loss: 2088403840.0000
Epoch 52/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 24ms/step - loss: 2088402048.0000
Epoch 53/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 24ms/step - loss: 2088400384.0000
Epoch 54/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 24ms/step - loss: 2088398336.0000
Epoch 55/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 24ms/step - loss: 2088396288.0000
Epoch 56/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 24ms/step - loss: 2088394496.0000
Epoch 57/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 24ms/step - loss: 2088392192.0000
Epoch 58/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 24ms/step - loss: 2088390016.0000
Epoch 59/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 24ms/step - loss: 2088387456.0000
Epoch 60/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 24ms/step - loss: 2088385152.0000
Epoch 61/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 24ms/step - loss: 2088382720.0000
Epoch 62/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 22ms/step - loss: 2088380160.0000
Epoch 63/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 24ms/step - loss: 2088377216.0000
Epoch 64/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 23ms/step - loss: 2088374528.0000
Epoch 65/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 24ms/step - loss: 2088371584.0000
Epoch 66/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 22ms/step - loss: 2088368640.0000
Epoch 67/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 23ms/step - loss: 2088365440.0000
Epoch 68/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 22ms/step - loss: 2088361984.0000
Epoch 69/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 24ms/step - loss: 2088358528.0000
Epoch 70/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 25ms/step - loss: 2088355328.0000
Epoch 71/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 24ms/step - loss: 2088351488.0000
Epoch 72/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 24ms/step - loss: 2088347648.0000
Epoch 73/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 24ms/step - loss: 2088343808.0000
Epoch 74/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 25ms/step - loss: 2088339456.0000
Epoch 75/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 22ms/step - loss: 2088335104.0000
Epoch 76/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 28ms/step - loss: 2088330752.0000
Epoch 77/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 24ms/step - loss: 2088326272.0000
Epoch 78/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 23ms/step - loss: 2088321536.0000
Epoch 79/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 24ms/step - loss: 2088316288.0000
Epoch 80/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 23ms/step - loss: 2088311424.0000
Epoch 81/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 23ms/step - loss: 2088306048.0000
Epoch 82/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 24ms/step - loss: 2088300800.0000
Epoch 83/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 22ms/step - loss: 2088295040.0000
Epoch 84/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 22ms/step - loss: 2088289408.0000
Epoch 85/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 24ms/step - loss: 2088283520.0000
Epoch 86/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 24ms/step - loss: 2088277120.0000
Epoch 87/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 24ms/step - loss: 2088270976.0000
Epoch 88/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 23ms/step - loss: 2088264192.0000
Epoch 89/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 22ms/step - loss: 2088257536.0000
Epoch 90/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 23ms/step - loss: 2088250880.0000
Epoch 91/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 24ms/step - loss: 2088243712.0000
Epoch 92/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 22ms/step - loss: 2088236416.0000
Epoch 93/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 23ms/step - loss: 2088229120.0000
Epoch 94/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 24ms/step - loss: 2088221568.0000
Epoch 95/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 23ms/step - loss: 2088213632.0000
Epoch 96/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 23ms/step - loss: 2088205440.0000
Epoch 97/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 21ms/step - loss: 2088197504.0000
Epoch 98/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 24ms/step - loss: 2088189312.0000
Epoch 99/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 28ms/step - loss: 2088180864.0000
Epoch 100/100
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 24ms/step - loss: 2088172032.0000
<keras.src.callbacks.history.History object at 0x00000295FF9CBD10>
>>>
>>> X_lstm_pred = scaler.inverse_transform(model_lstm.predict(X_lstm_test))
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 193ms/step
>>>
>>> # Plotting the results
>>> plt.figure(figsize=(15,6))
<Figure size 1500x600 with 0 Axes>
>>> plt.plot(y_test.values, color='red', label='Actual Ad Spend')
[<matplotlib.lines.Line2D object at 0x0000029587716840>]
>>> plt.plot(y_pred_lr_ad_spend, color='blue', label='LR Predicted Ad Spend (RMSE={:.2f})'.format(rmse_lr_ad_spend))
[<matplotlib.lines.Line2D object at 0x000002958A0EDB20>]
>>> plt.plot(y_pred_rf_ad_spend, color='green', label='RF Predicted Ad Spend (RMSE={:.2f})'.format(rmse_rf_ad_spend))
[<matplotlib.lines.Line2D object at 0x000002958A0EDDC0>]
>>> plt.plot(arima_ad_spend_pred, color='orange', label='ARIMA Predicted Ad Spend')
[<matplotlib.lines.Line2D object at 0x000002958A0EDFA0>]
>>> plt.plot(X_lstm_pred, color='purple', label='LSTM Predicted Ad Spend')
[<matplotlib.lines.Line2D object at 0x000002958A0EE2A0>]
>>> plt.title('Ad Spend Prediction')
Text(0.5, 1.0, 'Ad Spend Prediction')
>>> plt.xlabel('Index')
Text(0.5, 0, 'Index')
>>> plt.ylabel('Ad Spend')
Text(0, 0.5, 'Ad Spend')
>>> plt.legend()
<matplotlib.legend.Legend object at 0x000002958A0BFF80>
>>> plt.show()


>>> # Evaluate Linear Regression model
>>> rmse_lr_ad_spend = np.sqrt(mean_squared_error(y_test, y_pred_lr_ad_spend))
>>> print("Linear Regression RMSE:", rmse_lr_ad_spend)
Linear Regression RMSE: 1.8727642129282687e-11
>>>

>>> # Evaluate Random Forest Regression model
>>> rmse_rf_ad_spend = np.sqrt(mean_squared_error(y_test, y_pred_rf_ad_spend))
>>> print("Random Forest Regression RMSE:", rmse_rf_ad_spend)
Random Forest Regression RMSE: 9457.43924712005
>>>

>>>
>>> # Evaluate LSTM model
>>> lstm_rmse = np.sqrt(mean_squared_error(y_lstm_test, X_lstm_pred))
>>> print("LSTM RMSE:", lstm_rmse)
LSTM RMSE: 193039.03362562114

>>> # Evaluate ARIMA model
>>> arima_rmse = np.sqrt(mean_squared_error(df['Cost'].values[-6:], arima_ad_spend_pred))
>>> print("ARIMA RMSE:", arima_rmse)
ARIMA RMSE: 7420.462008336312
>>>