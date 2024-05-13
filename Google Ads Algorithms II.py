Python 3.12.3 (tags/v3.12.3:f6650f9, Apr  9 2024, 14:05:25) [MSC v.1938 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> import pandas as pd
>>> from sklearn.model_selection import train_test_split
>>> from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
>>> from sklearn.metrics import mean_squared_error
>>>
>>> # Load the data
>>> file_path = r"C:\Users\t14\Documents\Google Ads Data.xlsx"
>>> data = pd.read_excel(file_path)
>>>
>>> # Split the data into features (X) and target variables (y)
>>> X = data[['Clicks', 'Impr.', 'CTR', 'Avg. CPC', 'Cost', 'Conv. rate']]
>>> y_ctr = data['CTR']
>>> y_conversions = data['Conversions']
>>> y_cpc = data['Avg. CPC']
>>> y_conv_value = data['Conv. value']
>>>
>>> # Split the data into training and testing sets
>>> X_train, X_test, y_ctr_train, y_ctr_test, y_conversions_train, y_conversions_test, y_cpc_train, y_cpc_test, y_conv_value_train, y_conv_value_test = train_test_split(X, y_ctr, y_conversions, y_cpc, y_conv_value, test_size=0.2, random_state=42)
>>>
>>> # Train the classification model
>>> clf = RandomForestRegressor()  # RandomForestRegressor for regression
>>> clf.fit(X_train, y_ctr_train)
RandomForestRegressor()
>>> y_ctr_pred = clf.predict(X_test)
>>>
>>> # Train the regression model
>>> reg = GradientBoostingRegressor()
>>> reg.fit(X_train, y_cpc_train)
GradientBoostingRegressor()
>>> y_cpc_pred = reg.predict(X_test)
>>>
>>> # Evaluate the models
>>> ctr_rmse = mean_squared_error(y_ctr_test, y_ctr_pred, squared=False)
>>> cpc_rmse = mean_squared_error(y_cpc_test, y_cpc_pred, squared=False)

>>> # Print the results
>>> print(f"CTR model RMSE: {ctr_rmse}")
CTR model RMSE: 0.005447217317126199
>>> print(f"CPC model RMSE: {cpc_rmse}")
CPC model RMSE: 0.812319245439571
>>>