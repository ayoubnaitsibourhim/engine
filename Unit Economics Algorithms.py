>>> import pandas as pd
>>> from xgboost import XGBRegressor
>>> from sklearn.metrics import mean_squared_error
>>>
>>> # Define the data
>>> data = {
...     'Gross margin COGS': [65],
...     'Shipping cost': [130.00],
...     'Pick and Pack cost': [17.50],
...     'CAC 1': [124.04],
...     'Purchase frequency': [1.02],
...     'Average order value': [1172.20],
...     'Returns': [-5.39],
...     'Net Revenue': [1166.82],
...     'COGS': [-408.39],
...     'Shipping': [-130],
...     'Payment processing': [-19.69],
...     'Pick Pack': [-17.5],
...     'Total Costs': [-575.58],
...     'Gross margin': [50],
...     'Gross profit': [591.24],
...     'CAC as % of AOV': [11],
...     'Contribution margin': [40],
...     'Contribution per order': [467.19],
...     'CAC/CLV': [9.67],
...     'Contribution Profits': [467.19],
...     'Maximum CAC': [591.24],
...     'AOV': [1172.20],
...     'Breakeven ROAS': [1.982624055],
...     'Blended ROAS': [4.753552233],
...     'CLV': [1199.08]
... }
>>>
>>> # Create DataFrame
>>> df = pd.DataFrame(data)
>>>
>>> # Define features and target
>>> X = df.drop('CLV', axis=1)
>>> y = df['CLV']
>>>
>>> # Manually split the data into training and test sets
>>> X_train = X.iloc[:-1]
>>> y_train = y.iloc[:-1]
>>> X_test = X.iloc[-1:]
>>> y_test = y.iloc[-1:]
>>>
>>> # Train XGBoost model
>>> model = XGBRegressor()
>>> model.fit(X_train, y_train)

XGBRegressor(base_score=None, booster=None, callbacks=None,
             colsample_bylevel=None, colsample_bynode=None,
             colsample_bytree=None, device=None, early_stopping_rounds=None,
             enable_categorical=False, eval_metric=None, feature_types=None,
             gamma=None, grow_policy=None, importance_type=None,
             interaction_constraints=None, learning_rate=None, max_bin=None,
             max_cat_threshold=None, max_cat_to_onehot=None,
             max_delta_step=None, max_depth=None, max_leaves=None,
             min_child_weight=None, missing=nan, monotone_constraints=None,
             multi_strategy=None, n_estimators=None, n_jobs=None,
             num_parallel_tree=None, random_state=None, ...)
>>>
>>> # Make predictions
>>> y_pred = model.predict(X_test)
>>>
>>> # Calculate mean squared error
>>> mse = mean_squared_error(y_test, y_pred)
>>> print('Mean Squared Error:', mse)
Mean Squared Error: 1437792.8464



>>> # Increase AOV by 30% and set purchase frequency to 1.25
>>> new_aov = 1172.20 * 1.30
>>> purchase_frequency = 1.25
>>>
>>> # Calculate CLV
>>> predicted_clv = new_aov * purchase_frequency
>>>
>>> # Print the predicted CLV
>>> print('Predicted CLV with increased AOV and purchase frequency:', predicted_clv)
Predicted CLV with increased AOV and purchase frequency: 1904.8250000000003
>>> import matplotlib.pyplot as plt
>>>
>>> # Define the data
>>> labels = ['Current CLV', 'Predicted CLV']
>>> values = [1199.08, predicted_clv]
>>>
>>> # Create the bar plot
>>> plt.figure(figsize=(8, 6))
<Figure size 800x600 with 0 Axes>
>>> plt.bar(labels, values, color=['blue', 'orange'])
<BarContainer object of 2 artists>
>>> plt.ylabel('CLV')
Text(0, 0.5, 'CLV')
>>> plt.title('Current CLV vs Predicted CLV')
Text(0.5, 1.0, 'Current CLV vs Predicted CLV')
>>> plt.ylim(0, max(values) * 1.1)  # Set y-axis limit to show the full bar
(0.0, 2095.3075000000003)
>>> plt.show()
>>>



>>> import pandas as pd
>>> from sklearn.model_selection import train_test_split
>>> from sklearn.linear_model import Ridge
>>> from sklearn.metrics import mean_squared_error
>>>
>>> # Example DataFrame
>>> data = {
... 'Gross_Margin': [0.65, 0.70], # Example values
... 'Shipping_Cost': [130.00, 120.00],
... 'Pick_Pack_Cost': [17.50, 15.00],
... 'CAC': [124.04, 110.00],
... 'Purchase_Frequency': [1.02, 1.10],
... 'AOV': [1172.20, 1250.00],
... 'CLV': [1199.08, 1250.00] # Target variable
... }
>>>
>>> df = pd.DataFrame(data)
>>>
>>> # Splitting the dataset
>>> X = df.drop('CLV', axis=1)
>>> y = df['CLV']
>>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
>>>
>>> # Ridge Regression model
>>> ridge_model = Ridge(alpha=1.0)
>>> ridge_model.fit(X_train, y_train)
Ridge()
>>>
>>> # Predicting and evaluating the model
>>> y_pred = ridge_model.predict(X_test)
>>> mse = mean_squared_error(y_test, y_pred)
>>> print(f'Mean Squared Error: {mse}')
Mean Squared Error: 2592.8464000000076
>>>



>>> from sklearn.linear_model import LinearRegression
>>>
>>> # Assuming 'data' is already defined as shown above
>>> X = df[['Gross_Margin', 'Shipping_Cost', 'Pick_Pack_Cost', 'Purchase_Frequency', 'AOV']]
>>> y = df['CAC']
>>>
>>> # Splitting the dataset
>>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
>>>
>>> # Linear Regression model
>>> linear_model = LinearRegression()
>>> linear_model.fit(X_train, y_train)
LinearRegression()
>>>
>>> # Predicting and evaluating the model
>>> y_pred = linear_model.predict(X_test)
>>> mse = mean_squared_error(y_test, y_pred)
>>> print(f'Mean Squared Error: {mse}')
Mean Squared Error: 197.12160000000017
>>>

 # Based on the Mean Squared Errors obtained:
>>> # If a lower MSE is desired, target a lower CAC value. If an optimal balance is sought, aim for an optimal CAC value.
>>> # Visualizing the Mean Squared Errors for comparison.
>>> import matplotlib.pyplot as plt
>>>
>>> # Data for visualization
>>> models = ['Ridge Regression', 'Linear Regression']
>>> errors = [2592.8464000000076, 197.12160000000017]
>>>
>>> # Plotting
>>> plt.figure(figsize=(8, 5), facecolor='white')
<Figure size 800x500 with 0 Axes>
>>> plt.bar(models, errors, color=['blue', 'green'])
<BarContainer object of 2 artists>
>>> plt.xlabel('Regression Models')
Text(0.5, 0, 'Regression Models')
>>> plt.ylabel('Mean Squared Error')
Text(0, 0.5, 'Mean Squared Error')
>>> plt.title('Mean Squared Error Comparison')
Text(0.5, 1.0, 'Mean Squared Error Comparison')
>>> plt.show()

