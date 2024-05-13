>>> import pandas as pd
>>>
>>> # Load the Facebook Ads data directly from the Excel file
>>> facebook_ads_data = pd.read_excel("C:/Users/t14/OneDrive/Documents/Facebook Ads Data.xlsx")
>>>
>>> # Check the first few rows of the loaded data
>>> print(facebook_ads_data.head())
                                 Adset Name Date Start  ...  Adset Actions → Link Click  Ad Account → Name
0  (DV) ASC Performance // Evergreen Ad Set 2024-04-01  ...                       111.0              LuxUK
1  (DV) ASC Performance // Evergreen Ad Set 2024-04-01  ...                       107.0              LuxUK
2  (DV) ASC Performance // Evergreen Ad Set 2024-04-01  ...                       114.0              LuxUK
3  (DV) ASC Performance // Evergreen Ad Set 2024-04-01  ...                       129.0              LuxUK
4  (DV) ASC Performance // Evergreen Ad Set 2024-04-01  ...                       131.0              LuxUK

[5 rows x 7 columns]
>>> # Check for duplicates
>>> duplicates = facebook_ads_data.duplicated().sum()
>>>
>>> # Check for missing values
>>> missing_values = facebook_ads_data.isnull().sum()
>>>
>>> # Display the results
>>> print("Duplicates:", duplicates)
Duplicates: 0
>>> print("Missing Values:")
Missing Values:
>>> print(missing_values)
Adset Name                               0
Date Start                               0
Spend                                    9
Adset Actions → Purchase               474
Adset Action Values → Omni Purchase    432
Adset Actions → Link Click              18
Ad Account → Name                        0
dtype: int64
>>> # Calculate total spend
>>> total_spend = facebook_ads_data['Spend'].sum()
>>>
>>> # Calculate conversion rates
>>> conversion_rates = facebook_ads_data['Adset Actions → Purchase'] / facebook_ads_data['Adset Actions → Link Click']
>>>
>>> # Calculate CPC (Cost Per Click)
>>> cpc = facebook_ads_data['Spend'] / facebook_ads_data['Adset Actions → Link Click']
>>>
>>> # Display the results
>>> print("Total Spend:", total_spend)
Total Spend: 338574.55000000005
>>> print("Conversion Rates:")
Conversion Rates:
>>> print(conversion_rates)
0      0.009009
1           NaN
2      0.008772
3      0.007752
4      0.038168
         ...
720    0.020896
721         NaN
722         NaN
723    0.008591
724    0.006410
Length: 725, dtype: float64
>>> print("CPC:")
CPC:
>>> print(cpc)
0      2.446757
1      2.578785
2      2.433158
3      2.227752
4      2.201527
         ...
720    3.413433
721    3.825409
722    5.830678
723    1.399450
724    3.739167
Length: 725, dtype: float64
>>> from sklearn.linear_model import LinearRegression
>>> from sklearn.model_selection import train_test_split
>>> from sklearn.metrics import mean_squared_error, r2_score
>>>
>>> # Assuming your DataFrame is named 'facebook_ads_data'
>>> X = facebook_ads_data[['Adset Actions → Purchase', 'Adset Action Values → Omni Purchase', 'Adset Actions → Link Click']]
>>> y_ad_spend = facebook_ads_data['Spend']
>>>
>>> X_train, X_test, y_train, y_test = train_test_split(X, y_ad_spend, test_size=0.2, random_state=42)
>>>
>>> lr_ad_spend = LinearRegression()
>>> lr_ad_spend.fit(X_train, y_train)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "C:\Program Files\Python312\Lib\site-packages\sklearn\base.py", line 1474, in wrapper
    return fit_method(estimator, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Program Files\Python312\Lib\site-packages\sklearn\linear_model\_base.py", line 578, in fit
    X, y = self._validate_data(
           ^^^^^^^^^^^^^^^^^^^^
  File "C:\Program Files\Python312\Lib\site-packages\sklearn\base.py", line 650, in _validate_data
    X, y = check_X_y(X, y, **check_params)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Program Files\Python312\Lib\site-packages\sklearn\utils\validation.py", line 1263, in check_X_y
    X = check_array(
        ^^^^^^^^^^^^
  File "C:\Program Files\Python312\Lib\site-packages\sklearn\utils\validation.py", line 1049, in check_array
    _assert_all_finite(
  File "C:\Program Files\Python312\Lib\site-packages\sklearn\utils\validation.py", line 126, in _assert_all_finite
    _assert_all_finite_element_wise(
  File "C:\Program Files\Python312\Lib\site-packages\sklearn\utils\validation.py", line 175, in _assert_all_finite_element_wise
    raise ValueError(msg_err)
ValueError: Input X contains NaN.
LinearRegression does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is possible to preprocess the data, for instance by using an imputer transformer in a pipeline or drop samples with missing values. See https://scikit-learn.org/stable/modules/impute.html You can find a list of all estimators that handle NaN values at the following page: https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values
>>>
>>> y_pred = lr_ad_spend.predict(X_test)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "C:\Program Files\Python312\Lib\site-packages\sklearn\linear_model\_base.py", line 286, in predict
    return self._decision_function(X)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Program Files\Python312\Lib\site-packages\sklearn\linear_model\_base.py", line 269, in _decision_function
    X = self._validate_data(X, accept_sparse=["csr", "csc", "coo"], reset=False)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Program Files\Python312\Lib\site-packages\sklearn\base.py", line 633, in _validate_data
    out = check_array(X, input_name="X", **check_params)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Program Files\Python312\Lib\site-packages\sklearn\utils\validation.py", line 1049, in check_array
    _assert_all_finite(
  File "C:\Program Files\Python312\Lib\site-packages\sklearn\utils\validation.py", line 126, in _assert_all_finite
    _assert_all_finite_element_wise(
  File "C:\Program Files\Python312\Lib\site-packages\sklearn\utils\validation.py", line 175, in _assert_all_finite_element_wise
    raise ValueError(msg_err)
ValueError: Input X contains NaN.
LinearRegression does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is possible to preprocess the data, for instance by using an imputer transformer in a pipeline or drop samples with missing values. See https://scikit-learn.org/stable/modules/impute.html You can find a list of all estimators that handle NaN values at the following page: https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values
>>>
>>> # Evaluate the model
>>> print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'y_pred' is not defined
>>> print("R2 Score:", r2_score(y_test, y_pred))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'y_pred' is not defined
>>> from sklearn.impute import SimpleImputer
>>>
>>> # Assuming your DataFrame is named 'facebook_ads_data'
>>> X = facebook_ads_data[['Adset Actions → Purchase', 'Adset Action Values → Omni Purchase', 'Adset Actions → Link Click']]
>>> y_ad_spend = facebook_ads_data['Spend']
>>>
>>> # Impute missing values in X
>>> imputer = SimpleImputer(strategy='mean')
>>> X_imputed = imputer.fit_transform(X)
>>>
>>> X_train, X_test, y_train, y_test = train_test_split(X_imputed, y_ad_spend, test_size=0.2, random_state=42)
>>>
>>> lr_ad_spend = LinearRegression()
>>> lr_ad_spend.fit(X_train, y_train)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "C:\Program Files\Python312\Lib\site-packages\sklearn\base.py", line 1474, in wrapper
    return fit_method(estimator, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Program Files\Python312\Lib\site-packages\sklearn\linear_model\_base.py", line 578, in fit
    X, y = self._validate_data(
           ^^^^^^^^^^^^^^^^^^^^
  File "C:\Program Files\Python312\Lib\site-packages\sklearn\base.py", line 650, in _validate_data
    X, y = check_X_y(X, y, **check_params)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Program Files\Python312\Lib\site-packages\sklearn\utils\validation.py", line 1279, in check_X_y
    y = _check_y(y, multi_output=multi_output, y_numeric=y_numeric, estimator=estimator)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Program Files\Python312\Lib\site-packages\sklearn\utils\validation.py", line 1289, in _check_y
    y = check_array(
        ^^^^^^^^^^^^
  File "C:\Program Files\Python312\Lib\site-packages\sklearn\utils\validation.py", line 1049, in check_array
    _assert_all_finite(
  File "C:\Program Files\Python312\Lib\site-packages\sklearn\utils\validation.py", line 126, in _assert_all_finite
    _assert_all_finite_element_wise(
  File "C:\Program Files\Python312\Lib\site-packages\sklearn\utils\validation.py", line 175, in _assert_all_finite_element_wise
    raise ValueError(msg_err)
ValueError: Input y contains NaN.
>>>
>>> y_pred = lr_ad_spend.predict(X_test)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "C:\Program Files\Python312\Lib\site-packages\sklearn\linear_model\_base.py", line 286, in predict
    return self._decision_function(X)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Program Files\Python312\Lib\site-packages\sklearn\linear_model\_base.py", line 267, in _decision_function
    check_is_fitted(self)
  File "C:\Program Files\Python312\Lib\site-packages\sklearn\utils\validation.py", line 1622, in check_is_fitted
    raise NotFittedError(msg % {"name": type(estimator).__name__})
sklearn.exceptions.NotFittedError: This LinearRegression instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.
>>>
>>> # Evaluate the model
>>> print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'y_pred' is not defined
>>> print("R2 Score:", r2_score(y_test, y_pred))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'y_pred' is not defined
>>> missing_values = y_ad_spend.isnull().sum()
>>> if missing_values > 0:
...     print(f"There are {missing_values} missing values in y_ad_spend.")
...     # Handle missing values (e.g., impute or remove them)
...     # For example, to impute missing values with the mean:
...     y_ad_spend_imputed = y_ad_spend.fillna(y_ad_spend.mean())
... else:
...     y_ad_spend_imputed = y_ad_spend
...
There are 9 missing values in y_ad_spend.
>>> y_ad_spend_imputed = y_ad_spend.fillna(y_ad_spend.mean())
>>>
>>> X_train, X_test, y_train, y_test = train_test_split(X_imputed, y_ad_spend_imputed, test_size=0.2, random_state=42)
>>> lr_ad_spend = LinearRegression()
>>> lr_ad_spend.fit(X_train, y_train)
LinearRegression()
>>> y_pred = lr_ad_spend.predict(X_test)
>>>
>>> print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
Mean Squared Error: 349426.66002309474
>>> print("R2 Score:", r2_score(y_test, y_pred))
R2 Score: 0.04991857236265873




>>> from sklearn.ensemble import RandomForestRegressor
>>> rf_ad_spend = RandomForestRegressor(random_state=42)
>>> rf_ad_spend.fit(X_train, y_train)
RandomForestRegressor(random_state=42)
>>> y_pred_rf = rf_ad_spend.predict(X_test)
>>> print("Random Forest - Mean Squared Error:", mean_squared_error(y_test, y_pred_rf))
Random Forest - Mean Squared Error: 49531.04186988206
>>> print("Random Forest - R2 Score:", r2_score(y_test, y_pred_rf))
Random Forest - R2 Score: 0.8653264665924683


>>> facebook_ads_data['SMA_Adset_Actions_Purchase_3'] = facebook_ads_data['Adset Actions → Purchase'].rolling(window=3).mean()
>>> facebook_ads_data['SMA_Adset_Actions_Link_Click_5'] = facebook_ads_data['Adset Actions → Link Click'].rolling(window=5).mean()
>>> facebook_ads_data['SMA_Adset_Action_Values_Omni_Purchase_7'] = facebook_ads_data['Adset Action Values → Omni Purchase'].rolling(window=7).mean()


>>> import pandas as pd
>>> from statsmodels.tsa.arima.model import ARIMA
>>> from sklearn.model_selection import train_test_split
>>> from sklearn.preprocessing import MinMaxScaler
>>> from tensorflow.keras.models import Sequential
2024-05-10 18:43:58.692513: I tensorflow/core/util/port.cc:113] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2024-05-10 18:44:00.853355: I tensorflow/core/util/port.cc:113] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
>>> from tensorflow.keras.layers import LSTM, Dense
>>> from sklearn.metrics import mean_squared_error, r2_score
>>> data = facebook_ads_data[['Date Start', 'Spend', 'Adset Actions → Purchase', 'Adset Action Values → Omni Purchase', 'Adset Actions → Link Click']]
>>> data['Date Start'] = pd.to_datetime(data['Date Start'])
<stdin>:1: SettingWithCopyWarning:
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
>>> data.set_index('Date Start', inplace=True)
>>> train_size = int(len(data) * 0.8)
>>> train_data, test_data = data.iloc[:train_size], data.iloc[train_size:]
>>> scaler = MinMaxScaler()
>>> train_data_scaled = scaler.fit_transform(train_data)
>>> test_data_scaled = scaler.transform(test_data)
>>> model_arima = ARIMA(train_data['Spend'], order=(5,1,0))
C:\Program Files\Python312\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
C:\Program Files\Python312\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning: A date index has been provided, but it is not monotonic and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
C:\Program Files\Python312\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
C:\Program Files\Python312\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning: A date index has been provided, but it is not monotonic and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
C:\Program Files\Python312\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
C:\Program Files\Python312\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:473: ValueWarning: A date index has been provided, but it is not monotonic and so will be ignored when e.g. forecasting.
  self._init_dates(dates, freq)
>>> model_fit = model_arima.fit()
>>> def create_dataset(X, y, time_steps=1):
...     Xs, ys = [], []
...     for i in range(len(X) - time_steps):
...         v = X.iloc[i:(i + time_steps)].values
...         Xs.append(v)
...         ys.append(y.iloc[i + time_steps])
...     return np.array(Xs), np.array(ys)
... TIME_STEPS = 10
  File "<stdin>", line 8
    TIME_STEPS = 10
    ^^^^^^^^^^
SyntaxError: invalid syntax
>>> X_train, y_train = create_dataset(train_data_scaled, train_data_scaled['Spend'], TIME_STEPS)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'create_dataset' is not defined
>>> X_test, y_test = create_dataset(test_data_scaled, test_data_scaled['Spend'], TIME_STEPS)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'create_dataset' is not defined
>>> model_lstm = Sequential()
>>> model_lstm.add(LSTM(units=50, input_shape=(X_train.shape[1], X_train.shape[2])))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
IndexError: tuple index out of range
>>> model_lstm.add(Dense(1))
>>> model_lstm.compile(loss='mean_squared_error', optimizer='adam')
2024-05-10 18:46:12.867316: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
>>> model_lstm.fit(X_train, y_train, epochs=100, batch_size=32)
Epoch 1/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 741us/step - loss: 35471084.0000
Epoch 2/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 461us/step - loss: 40197188.0000
Epoch 3/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 452us/step - loss: 34329716.0000
Epoch 4/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 443us/step - loss: 33522702.0000
Epoch 5/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 421us/step - loss: 64328440.0000
Epoch 6/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 500us/step - loss: 40697852.0000
Epoch 7/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 453us/step - loss: 29837498.0000
Epoch 8/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 493us/step - loss: 26231250.0000
Epoch 9/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 459us/step - loss: 39284032.0000
Epoch 10/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 512us/step - loss: 60035024.0000
Epoch 11/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 444us/step - loss: 35552040.0000
Epoch 12/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 445us/step - loss: 50896932.0000
Epoch 13/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 451us/step - loss: 27914020.0000
Epoch 14/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 444us/step - loss: 37915152.0000
Epoch 15/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 445us/step - loss: 29489266.0000
Epoch 16/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 444us/step - loss: 39429568.0000
Epoch 17/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 452us/step - loss: 32340266.0000
Epoch 18/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 492us/step - loss: 20483508.0000
Epoch 19/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 519us/step - loss: 25126836.0000
Epoch 20/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 444us/step - loss: 19736848.0000
Epoch 21/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 427us/step - loss: 19332728.0000
Epoch 22/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 444us/step - loss: 17916346.0000
Epoch 23/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 444us/step - loss: 27255162.0000
Epoch 24/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 430us/step - loss: 41099628.0000
Epoch 25/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 395us/step - loss: 59477748.0000
Epoch 26/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 460us/step - loss: 21020482.0000
Epoch 27/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 437us/step - loss: 21882306.0000
Epoch 28/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 501us/step - loss: 36397092.0000
Epoch 29/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 500us/step - loss: 20176980.0000
Epoch 30/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 521us/step - loss: 51355440.0000
Epoch 31/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 454us/step - loss: 19532118.0000
Epoch 32/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 437us/step - loss: 19039394.0000
Epoch 33/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 453us/step - loss: 16304760.0000
Epoch 34/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 444us/step - loss: 17031190.0000
Epoch 35/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 444us/step - loss: 13955721.0000
Epoch 36/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 395us/step - loss: 15823951.0000
Epoch 37/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 431us/step - loss: 13689760.0000
Epoch 38/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 501us/step - loss: 16330015.0000
Epoch 39/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 445us/step - loss: 12352554.0000
Epoch 40/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 443us/step - loss: 17004564.0000
Epoch 41/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 444us/step - loss: 24185198.0000
Epoch 42/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 516us/step - loss: 12722694.0000
Epoch 43/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 443us/step - loss: 10838567.0000
Epoch 44/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 459us/step - loss: 32541594.0000
Epoch 45/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 442us/step - loss: 21878640.0000
Epoch 46/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 441us/step - loss: 14585727.0000
Epoch 47/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 444us/step - loss: 16214473.0000
Epoch 48/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 443us/step - loss: 12043424.0000
Epoch 49/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 444us/step - loss: 28003070.0000
Epoch 50/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 443us/step - loss: 13782361.0000
Epoch 51/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 444us/step - loss: 14044084.0000
Epoch 52/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 500us/step - loss: 11108986.0000
Epoch 53/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 444us/step - loss: 13355905.0000
Epoch 54/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 536us/step - loss: 16291221.0000
Epoch 55/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 451us/step - loss: 11072888.0000
Epoch 56/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 500us/step - loss: 9846504.0000
Epoch 57/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 473us/step - loss: 9096110.0000
Epoch 58/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 444us/step - loss: 8065887.5000
Epoch 59/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 494us/step - loss: 11359492.0000
Epoch 60/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 457us/step - loss: 14611044.0000
Epoch 61/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 527us/step - loss: 10753806.0000
Epoch 62/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 452us/step - loss: 9662990.0000
Epoch 63/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 492us/step - loss: 9103523.0000
Epoch 64/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 445us/step - loss: 7970174.0000
Epoch 65/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 445us/step - loss: 7967764.5000
Epoch 66/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 520us/step - loss: 6188197.5000
Epoch 67/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 443us/step - loss: 7146149.5000
Epoch 68/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 445us/step - loss: 11667916.0000
Epoch 69/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 444us/step - loss: 10526914.0000
Epoch 70/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 436us/step - loss: 7582012.0000
Epoch 71/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 473us/step - loss: 6174252.0000
Epoch 72/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 444us/step - loss: 6169081.5000
Epoch 73/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 535us/step - loss: 6230783.0000
Epoch 74/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 469us/step - loss: 5697482.5000
Epoch 75/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 451us/step - loss: 4656041.0000
Epoch 76/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 445us/step - loss: 4768468.0000
Epoch 77/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 417us/step - loss: 4671232.5000
Epoch 78/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 451us/step - loss: 5549765.0000
Epoch 79/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 464us/step - loss: 8925751.0000
Epoch 80/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 494us/step - loss: 4476201.5000
Epoch 81/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 537us/step - loss: 4111153.7500
Epoch 82/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 542us/step - loss: 4406400.0000
Epoch 83/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 389us/step - loss: 5860035.5000
Epoch 84/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 508us/step - loss: 4595904.0000
Epoch 85/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 481us/step - loss: 3535354.2500
Epoch 86/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 445us/step - loss: 4417787.0000
Epoch 87/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 500us/step - loss: 3127424.0000
Epoch 88/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 445us/step - loss: 6103247.0000
Epoch 89/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 536us/step - loss: 3352919.0000
Epoch 90/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 537us/step - loss: 3158960.0000
Epoch 91/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 444us/step - loss: 5375999.0000
Epoch 92/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 445us/step - loss: 3903756.7500
Epoch 93/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 448us/step - loss: 3001980.7500
Epoch 94/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 444us/step - loss: 3129697.7500
Epoch 95/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 445us/step - loss: 4876067.5000
Epoch 96/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 500us/step - loss: 2686455.7500
Epoch 97/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 444us/step - loss: 2887700.7500
Epoch 98/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 444us/step - loss: 2787667.0000
Epoch 99/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 536us/step - loss: 2695630.5000
Epoch 100/100
←[1m19/19←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 444us/step - loss: 2816293.7500
<keras.src.callbacks.history.History object at 0x000001F75C517410>
>>> predictions_arima = model_fit.forecast(steps=len(test_data))
C:\Program Files\Python312\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:836: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
  return get_prediction_index(
C:\Program Files\Python312\Lib\site-packages\statsmodels\tsa\base\tsa_model.py:836: FutureWarning: No supported index is available. In the next version, calling this method in a model without a supported index will result in an exception.
  return get_prediction_index(
>>> predictions_lstm = model_lstm.predict(X_test)
←[1m5/5←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 2ms/step
>>> predictions_arima = scaler.inverse_transform(predictions_arima.reshape(-1, 1)).flatten()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "C:\Program Files\Python312\Lib\site-packages\pandas\core\generic.py", line 6299, in __getattr__
    return object.__getattribute__(self, name)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'Series' object has no attribute 'reshape'. Did you mean: 'shape'?
>>> predictions_lstm = scaler.inverse_transform(predictions_lstm).flatten()
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "C:\Program Files\Python312\Lib\site-packages\sklearn\preprocessing\_data.py", line 572, in inverse_transform
    X -= self.min_
ValueError: non-broadcastable output operand with shape (145,1) doesn't match the broadcast shape (145,4)
>>> print("ARIMA - Mean Squared Error:", mean_squared_error(test_data['Spend'], predictions_arima))
ARIMA - Mean Squared Error: 2534861.1460629627
>>> print("ARIMA - R2 Score:", r2_score(test_data['Spend'], predictions_arima))
ARIMA - R2 Score: -0.12784716829039966
>>> print("LSTM - Mean Squared Error:", mean_squared_error(test_data['Spend'], predictions_lstm))
LSTM - Mean Squared Error: 4233181.441241874
>>> print("LSTM - R2 Score:", r2_score(test_data['Spend'], predictions_lstm))
LSTM - R2 Score: -0.8834884541030918
>>>
>>>






>>> import pandas as pd
>>> google_ads_data = pd.read_excel(r"C:\Users\t14\OneDrive\Documents\Google Ads Data.xlsx")
>>> print(google_ads_data.head())
       Month  Clicks  Impr.     CTR Currency code  ...  View-through conv.  Cost / conv.  Conv. rate  All conv.  Conv. value
0 2023-01-01    1564  15443  0.1013           DKK  ...                   0        213.39      0.0254      87.77     41026.08
1 2023-02-01    2019  14427  0.1399           DKK  ...                   0         78.24      0.0490      98.92    104279.44
2 2023-03-01    2713  17835  0.1521           DKK  ...                   0        144.82      0.0209      56.78     62696.71
3 2023-04-01    3093  22619  0.1367           DKK  ...                   0        159.56      0.0184      58.17     69320.88
4 2024-04-01     207   6650  0.0311           DKK  ...                   0       2391.08      0.0048       2.00      1377.45

[5 rows x 15 columns]
>>> X_google_ads = google_ads_data[['Clicks', 'Impr.', 'CTR', 'Avg. CPC', 'Cost', 'Impr. (Abs. Top) %', 'Impr. (Top) %', 'View-through conv.', 'Cost / conv.', 'Conv. rate', 'All conv.']]
>>> y_ctr = google_ads_data['CTR']
>>> y_conversions = google_ads_data['Conversions']
>>> y_cpc_prices = google_ads_data['Avg. CPC']
>>> y_conversion_values = google_ads_data['Conv. value']
>>> X_ctr_train, X_ctr_test, y_ctr_train, y_ctr_test = train_test_split(X_google_ads, y_ctr, test_size=0.2, random_state=42)
>>> X_conversions_train, X_conversions_test, y_conversions_train, y_conversions_test = train_test_split(X_google_ads, y_conversions, test_size=0.2, random_state=42)
>>> X_cpc_prices_train, X_cpc_prices_test, y_cpc_prices_train, y_cpc_prices_test = train_test_split(X_google_ads, y_cpc_prices, test_size=0.2, random_state=42)
>>> X_conversion_values_train, X_conversion_values_test, y_conversion_values_train, y_conversion_values_test = train_test_split(X_google_ads, y_conversion_values, test_size=0.2, random_state=42)
>>> from sklearn.linear_model import LogisticRegression
>>> from sklearn.ensemble import RandomForestClassifier
>>> lr_ctr = LogisticRegression()
>>> rf_ctr = RandomForestClassifier(random_state=42)
>>> lr_ctr.fit(X_ctr_train, y_ctr_train)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "C:\Program Files\Python312\Lib\site-packages\sklearn\base.py", line 1474, in wrapper
    return fit_method(estimator, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Program Files\Python312\Lib\site-packages\sklearn\linear_model\_logistic.py", line 1209, in fit
    check_classification_targets(y)
  File "C:\Program Files\Python312\Lib\site-packages\sklearn\utils\multiclass.py", line 221, in check_classification_targets
    raise ValueError(
ValueError: Unknown label type: continuous. Maybe you are trying to fit a classifier, which expects discrete classes on a regression target with continuous values.
>>> rf_ctr.fit(X_ctr_train, y_ctr_train)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "C:\Program Files\Python312\Lib\site-packages\sklearn\base.py", line 1474, in wrapper
    return fit_method(estimator, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Program Files\Python312\Lib\site-packages\sklearn\ensemble\_forest.py", line 421, in fit
    y, expanded_class_weight = self._validate_y_class_weight(y)
                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Program Files\Python312\Lib\site-packages\sklearn\ensemble\_forest.py", line 831, in _validate_y_class_weight
    check_classification_targets(y)
  File "C:\Program Files\Python312\Lib\site-packages\sklearn\utils\multiclass.py", line 221, in check_classification_targets
    raise ValueError(
ValueError: Unknown label type: continuous. Maybe you are trying to fit a classifier, which expects discrete classes on a regression target with continuous values.
>>> from sklearn.linear_model import LinearRegression
>>> from sklearn.ensemble import GradientBoostingRegressor
>>>
>>> lr_cpc_prices = LinearRegression()
>>> gb_cpc_prices = GradientBoostingRegressor(random_state=42)
>>>
>>> lr_cpc_prices.fit(X_cpc_prices_train, y_cpc_prices_train)
LinearRegression()
>>> gb_cpc_prices.fit(X_cpc_prices_train, y_cpc_prices_train)
GradientBoostingRegressor(random_state=42)
>>>
>>> # Implementing the Regression Model (Linear Regression or Gradient Boosting) for Conversion values
>>> lr_conversion_values = LinearRegression()
>>> gb_conversion_values = GradientBoostingRegressor(random_state=42)
>>>
>>> lr_conversion_values.fit(X_conversion_values_train, y_conversion_values_train)
LinearRegression()
>>> gb_conversion_values.fit(X_conversion_values_train, y_conversion_values_train)
GradientBoostingRegressor(random_state=42)
>>> from sklearn.metrics import classification_report
>>> y_pred_class = classifier.predict(X_test)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'classifier' is not defined
>>> from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
>>> lr_ctr_score = lr_ctr.score(X_ctr_test, y_ctr_test)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "C:\Program Files\Python312\Lib\site-packages\sklearn\base.py", line 764, in score
    return accuracy_score(y, self.predict(X), sample_weight=sample_weight)
                             ^^^^^^^^^^^^^^^
  File "C:\Program Files\Python312\Lib\site-packages\sklearn\linear_model\_base.py", line 351, in predict
    scores = self.decision_function(X)
             ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Program Files\Python312\Lib\site-packages\sklearn\linear_model\_base.py", line 333, in decision_function
    scores = safe_sparse_dot(X, self.coef_.T, dense_output=True) + self.intercept_
                                ^^^^^^^^^^
AttributeError: 'LogisticRegression' object has no attribute 'coef_'
>>> rf_ctr_score = rf_ctr.score(X_ctr_test, y_ctr_test)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "C:\Program Files\Python312\Lib\site-packages\sklearn\base.py", line 764, in score
    return accuracy_score(y, self.predict(X), sample_weight=sample_weight)
                             ^^^^^^^^^^^^^^^
  File "C:\Program Files\Python312\Lib\site-packages\sklearn\ensemble\_forest.py", line 905, in predict
    proba = self.predict_proba(X)
            ^^^^^^^^^^^^^^^^^^^^^
  File "C:\Program Files\Python312\Lib\site-packages\sklearn\ensemble\_forest.py", line 947, in predict_proba
    X = self._validate_X_predict(X)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Program Files\Python312\Lib\site-packages\sklearn\ensemble\_forest.py", line 636, in _validate_X_predict
    if self.estimators_[0]._support_missing_values(X):
       ^^^^^^^^^^^^^^^^
AttributeError: 'RandomForestClassifier' object has no attribute 'estimators_'. Did you mean: 'estimator'?
>>> print("Logistic Regression - Accuracy Score for CTR:", lr_ctr_score)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'lr_ctr_score' is not defined
>>> print("Random Forest Classifier - Accuracy Score for CTR:", rf_ctr_score)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'rf_ctr_score' is not defined
>>> from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
>>>
>>> lr_ctr_score = lr_ctr.score(X_ctr_test, y_ctr_test)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "C:\Program Files\Python312\Lib\site-packages\sklearn\base.py", line 764, in score
    return accuracy_score(y, self.predict(X), sample_weight=sample_weight)
                             ^^^^^^^^^^^^^^^
  File "C:\Program Files\Python312\Lib\site-packages\sklearn\linear_model\_base.py", line 351, in predict
    scores = self.decision_function(X)
             ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Program Files\Python312\Lib\site-packages\sklearn\linear_model\_base.py", line 333, in decision_function
    scores = safe_sparse_dot(X, self.coef_.T, dense_output=True) + self.intercept_
                                ^^^^^^^^^^
AttributeError: 'LogisticRegression' object has no attribute 'coef_'
>>> rf_ctr_score = rf_ctr.score(X_ctr_test, y_ctr_test)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "C:\Program Files\Python312\Lib\site-packages\sklearn\base.py", line 764, in score
    return accuracy_score(y, self.predict(X), sample_weight=sample_weight)
                             ^^^^^^^^^^^^^^^
  File "C:\Program Files\Python312\Lib\site-packages\sklearn\ensemble\_forest.py", line 905, in predict
    proba = self.predict_proba(X)
            ^^^^^^^^^^^^^^^^^^^^^
  File "C:\Program Files\Python312\Lib\site-packages\sklearn\ensemble\_forest.py", line 947, in predict_proba
    X = self._validate_X_predict(X)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Program Files\Python312\Lib\site-packages\sklearn\ensemble\_forest.py", line 636, in _validate_X_predict
    if self.estimators_[0]._support_missing_values(X):
       ^^^^^^^^^^^^^^^^
AttributeError: 'RandomForestClassifier' object has no attribute 'estimators_'. Did you mean: 'estimator'?
>>>
>>> print("Logistic Regression - Accuracy Score for CTR:", lr_ctr_score)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'lr_ctr_score' is not defined
>>> print("Random Forest Classifier - Accuracy Score for CTR:", rf_ctr_score)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'rf_ctr_score' is not defined
>>> from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
>>>
>>> lr_ctr_pred = lr_ctr.predict(X_ctr_test)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "C:\Program Files\Python312\Lib\site-packages\sklearn\linear_model\_base.py", line 351, in predict
    scores = self.decision_function(X)
             ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Program Files\Python312\Lib\site-packages\sklearn\linear_model\_base.py", line 333, in decision_function
    scores = safe_sparse_dot(X, self.coef_.T, dense_output=True) + self.intercept_
                                ^^^^^^^^^^
AttributeError: 'LogisticRegression' object has no attribute 'coef_'
>>> rf_ctr_pred = rf_ctr.predict(X_ctr_test)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "C:\Program Files\Python312\Lib\site-packages\sklearn\ensemble\_forest.py", line 905, in predict
    proba = self.predict_proba(X)
            ^^^^^^^^^^^^^^^^^^^^^
  File "C:\Program Files\Python312\Lib\site-packages\sklearn\ensemble\_forest.py", line 947, in predict_proba
    X = self._validate_X_predict(X)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Program Files\Python312\Lib\site-packages\sklearn\ensemble\_forest.py", line 636, in _validate_X_predict
    if self.estimators_[0]._support_missing_values(X):
       ^^^^^^^^^^^^^^^^
AttributeError: 'RandomForestClassifier' object has no attribute 'estimators_'. Did you mean: 'estimator'?
>>> from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
>>>
>>> lr_ctr_pred = lr_ctr.predict(X_ctr_test)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "C:\Program Files\Python312\Lib\site-packages\sklearn\linear_model\_base.py", line 351, in predict
    scores = self.decision_function(X)
             ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Program Files\Python312\Lib\site-packages\sklearn\linear_model\_base.py", line 333, in decision_function
    scores = safe_sparse_dot(X, self.coef_.T, dense_output=True) + self.intercept_
                                ^^^^^^^^^^
AttributeError: 'LogisticRegression' object has no attribute 'coef_'
>>> rf_ctr_pred = rf_ctr.predict(X_ctr_test)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "C:\Program Files\Python312\Lib\site-packages\sklearn\ensemble\_forest.py", line 905, in predict
    proba = self.predict_proba(X)
            ^^^^^^^^^^^^^^^^^^^^^
  File "C:\Program Files\Python312\Lib\site-packages\sklearn\ensemble\_forest.py", line 947, in predict_proba
    X = self._validate_X_predict(X)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Program Files\Python312\Lib\site-packages\sklearn\ensemble\_forest.py", line 636, in _validate_X_predict
    if self.estimators_[0]._support_missing_values(X):
       ^^^^^^^^^^^^^^^^
AttributeError: 'RandomForestClassifier' object has no attribute 'estimators_'. Did you mean: 'estimator'?
>>> y_ctr_test_bin = (y_ctr_test > y_ctr_test.mean()).astype(int)
>>> lr_ctr_pred = lr_ctr.predict(X_ctr_test)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "C:\Program Files\Python312\Lib\site-packages\sklearn\linear_model\_base.py", line 351, in predict
    scores = self.decision_function(X)
             ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Program Files\Python312\Lib\site-packages\sklearn\linear_model\_base.py", line 333, in decision_function
    scores = safe_sparse_dot(X, self.coef_.T, dense_output=True) + self.intercept_
                                ^^^^^^^^^^
AttributeError: 'LogisticRegression' object has no attribute 'coef_'
>>> rf_ctr_pred = rf_ctr.predict(X_ctr_test)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "C:\Program Files\Python312\Lib\site-packages\sklearn\ensemble\_forest.py", line 905, in predict
    proba = self.predict_proba(X)
            ^^^^^^^^^^^^^^^^^^^^^
  File "C:\Program Files\Python312\Lib\site-packages\sklearn\ensemble\_forest.py", line 947, in predict_proba
    X = self._validate_X_predict(X)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Program Files\Python312\Lib\site-packages\sklearn\ensemble\_forest.py", line 636, in _validate_X_predict
    if self.estimators_[0]._support_missing_values(X):
       ^^^^^^^^^^^^^^^^
AttributeError: 'RandomForestClassifier' object has no attribute 'estimators_'. Did you mean: 'estimator'?
>>> lr_ctr_score = accuracy_score(y_ctr_test_bin, lr_ctr_pred)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'lr_ctr_pred' is not defined
>>> rf_ctr_score = accuracy_score(y_ctr_test_bin, rf_ctr_pred)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'rf_ctr_pred' is not defined
>>> print("Logistic Regression - Accuracy Score for CTR:", lr_ctr_score)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'lr_ctr_score' is not defined
>>> print("Random Forest Classifier - Accuracy Score for CTR:", rf_ctr_score)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'rf_ctr_score' is not defined