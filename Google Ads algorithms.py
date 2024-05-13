>>> import pandas as pd
>>> data = {
...     'Month': ['Jan-23', 'Feb-23', 'Mar-23', 'Apr-23', 'Apr-24', 'Mar-24', 'Feb-24', 'Jan-24', 'Dec-23', 'Nov-23', 'Oct-23', 'Sep-23', 'Aug-23', 'Jul-23', 'Jun-23', 'May-23'],
...     'Clicks': [1564, 2019, 2713, 3093, 207, 6577, 7392, 12153, 9725, 9083, 5965, 6561, 4918, 3607, 3423, 4105],
...     'Impr.': [15443, 14427, 17835, 22619, 6650, 238346, 361321, 439276, 200840, 163894, 63508, 176881, 95395, 58312, 61112, 82835],
...     'CTR': ['10.13%', '13.99%', '15.21%', '13.67%', '3.11%', '2.76%', '2.05%', '2.77%', '4.84%', '5.54%', '9.39%', '3.71%', '5.16%', '6.19%', '5.60%', '4.96%'],
...     'Currency code': ['DKK', 'DKK', 'DKK', 'DKK', 'DKK', 'DKK', 'DKK', 'DKK', 'DKK', 'DKK', 'DKK', 'DKK', 'DKK', 'DKK', 'DKK', 'DKK'],
...     'Avg. CPC': [5.43, 3.83, 3.03, 2.94, 11.55, 8.49, 6.58, 5.78, 6.58, 7.17, 5.81, 7.32, 5.93, 6.11, 6.42, 4.94],
...     'Cost': [8485.77, 7739.13, 8223.6, 9097.88, 2391.08, 55868.81, 48622.2, 70220.19, 63988.05, 65084.98, 34657.49, 48050.65, 29160.95, 22038.46, 21961.18, 20275.8],
...     'Impr. (Abs. Top) %': ['63.45%', '74.59%', '77.88%', '52.16%', '38.71%', '36.30%', '34.37%', '34.03%', '42.06%', '44.92%', '48.86%', '59.58%', '52.29%', '55.19%', '51.23%', '53.33%'],
...     'Impr. (Top) %': ['80.19%', '86.63%', '84.57%', '76.73%', '81.37%', '79.11%', '83.71%', '81.18%', '83.62%', '86.86%', '88.68%', '88.60%', '85.85%', '86.92%', '86.19%', '85.62%'],
...     'Conversions': [39.77, 98.92, 56.78, 57.02, 1, 119.97, 173.79, 308.86, 326.15, 389.12, 203.87, 239.09, 146.93, 71.13, 142.8, 145.51],
...     'View-through conv.': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
...     'Cost / conv.': [213.39, 78.24, 144.82, 159.56, 2391.08, 465.69, 279.78, 227.35, 196.19, 167.26, 170, 200.98, 198.47, 309.85, 153.79, 139.34],
...     'Conv. rate': ['2.54%', '4.90%', '2.09%', '1.84%', '0.48%', '1.77%', '0.61%', '0.53%', '3.06%', '3.81%', '3.24%', '3.26%', '2.99%', '1.97%', '4.17%', '3.54%'],
...     'All conv.': [87.77, 98.92, 56.78, 58.17, 2, 248.94, 323.44, 588.45, 617.75, 759.39, 384.77, 331.2, 150.93, 71.13, 336.61, 556.56],
...     'Conv. value': [41026.08, 104279.44, 62696.71, 69320.88, 1377.45, 217817.05, 198222.21, 351147.61, 367925.39, 482623.12, 232522.95, 274986.65, 178724.91, 82735.07, 178625.55, 173590.2]
... }
>>> google_ads_data = pd.DataFrame(data)
>>> from sklearn.model_selection import train_test_split
>>> from sklearn.linear_model import LogisticRegression
>>> from sklearn.ensemble import RandomForestClassifier
>>> from sklearn.metrics import accuracy_score
>>> X_ctr_train, X_ctr_test, y_ctr_train, y_ctr_test = train_test_split(google_ads_data[['Clicks', 'Impr.', 'Avg. CPC', 'Cost', 'Conversions']], google_ads_data['CTR'], test_size=0.2, random_state=42)
>>> lr_ctr = LogisticRegression()
>>> rf_ctr = RandomForestClassifier(random_state=42)
>>>
>>> lr_ctr.fit(X_ctr_train, y_ctr_train)
C:\Program Files\Python312\Lib\site-packages\sklearn\linear_model\_logistic.py:469: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
LogisticRegression()
>>> rf_ctr.fit(X_ctr_train, y_ctr_train)
RandomForestClassifier(random_state=42)
>>>
>>> lr_ctr_score = lr_ctr.score(X_ctr_test, y_ctr_test)
>>> rf_ctr_score = rf_ctr.score(X_ctr_test, y_ctr_test)
>>> from sklearn.linear_model import LinearRegression
>>> from sklearn.ensemble import GradientBoostingRegressor
>>> from sklearn.metrics import r2_score
>>>
>>> X_conv_train, X_conv_test, y_conv_train, y_conv_test = train_test_split(google_ads_data[['Clicks', 'Impr.', 'Avg. CPC', 'Cost', 'Conversions']], google_ads_data['Conv. value'], test_size=0.2, random_state=42)
>>> lr_conv = LinearRegression()
>>> gb_conv = GradientBoostingRegressor(random_state=42)
>>>
>>> lr_conv.fit(X_conv_train, y_conv_train)
LinearRegression()
>>> gb_conv.fit(X_conv_train, y_conv_train)
GradientBoostingRegressor(random_state=42)
>>>
>>> lr_conv_score = lr_conv.score(X_conv_test, y_conv_test)
>>> gb_conv_score = gb_conv.score(X_conv_test, y_conv_test)
>>>
>>> lr_ctr_score, rf_ctr_score, lr_conv_score, gb_conv_score
(0.0, 0.0, 0.4325955414092265, 0.8712819775215734)
>>>