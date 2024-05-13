>>> import pandas as pd
>>> from scipy.stats import chi2_contingency
>>>
>>> # CRO data
>>> data = {
...     'DEVICE/STAGE': ['MOBILE', 'DESKTOP', 'TABLET', 'TOTAL'],
...     'Page View': [7869, 684, 87, 8640],
...     'View Item': [6066, 300, 46, 6412],
...     'Add To Cart': [695, 47, 6, 748],
...     'Purchase': [344, 19, 4, 367]
... }
>>>
>>> df = pd.DataFrame(data)
>>>
>>> # Set 'DEVICE/STAGE' as index
>>> df.set_index('DEVICE/STAGE', inplace=True)
>>>
>>> # Perform chi-squared test
>>> chi2, p, _, _ = chi2_contingency(df)
>>> print(f"Chi-squared value: {chi2}, p-value: {p}")
Chi-squared value: 69.21316225483295, p-value: 2.1717166583523457e-11
>>>



>>> from sklearn.tree import DecisionTreeClassifier
>>> from sklearn.model_selection import train_test_split
>>> from sklearn.metrics import accuracy_score
>>>
>>> # CRO data for decision tree
>>> data = {
...     'DEVICE/STAGE': ['MOBILE', 'DESKTOP', 'TABLET', 'TOTAL'],
...     'Page View': [7869, 684, 87, 8640],
...     'View Item': [6066, 300, 46, 6412],
...     'Add To Cart': [695, 47, 6, 748],
...     'Purchase': [344, 19, 4, 367]
... }
>>>
>>> df = pd.DataFrame(data)
>>>
>>> # Prepare the data
>>> X = df[['Page View', 'View Item', 'Add To Cart']]
>>> y = df['Purchase']
>>>
>>> # Split the data into training and testing sets
>>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
>>>
>>> # Fit the decision tree model
>>> model = DecisionTreeClassifier(random_state=42)
>>> model.fit(X_train, y_train)
DecisionTreeClassifier(random_state=42)
>>>
>>> # Make predictions
>>> y_pred = model.predict(X_test)
>>>


>>> import pandas as pd
>>> import numpy as np
>>> from scipy.stats import chi2_contingency
>>>
>>> # CRO data
>>> data = {
...     'Ga:shoppingStage': ['add_to_cart', 'add_to_cart', 'add_to_cart', 'page_view', 'page_view', 'page_view', 'purchase', 'purchase', 'purchase', 'view_item', 'view_item', 'view_item'],
...     'Ga:deviceCategory': ['mobile', 'mobile', 'mobile', 'mobile', 'mobile', 'mobile', 'mobile', 'mobile', 'mobile', 'mobile', 'mobile', 'mobile'],
...     'Ga:month': [202402, 202403, 202404, 202402, 202403, 202404, 202402, 202403, 202404, 202402, 202403, 202404],
...     'Sum of Users': [518, 113, 64, 6242, 1073, 554, 287, 36, 21, 5140, 678, 248]
... }
>>>
>>> df = pd.DataFrame(data)
>>>
>>> # Display the DataFrame
>>> print(df)
   Ga:shoppingStage Ga:deviceCategory  Ga:month  Sum of Users
0       add_to_cart            mobile    202402           518
1       add_to_cart            mobile    202403           113
2       add_to_cart            mobile    202404            64
3         page_view            mobile    202402          6242
4         page_view            mobile    202403          1073
5         page_view            mobile    202404           554
6          purchase            mobile    202402           287
7          purchase            mobile    202403            36
8          purchase            mobile    202404            21
9         view_item            mobile    202402          5140
10        view_item            mobile    202403           678
11        view_item            mobile    202404           248
>>> # Create contingency table
>>> contingency_table = pd.pivot_table(df, values='Sum of Users', index='Ga:shoppingStage', columns='Ga:deviceCategory', aggfunc='sum')
>>>
>>> # Perform chi-squared test
>>> chi2, p, _, _ = chi2_contingency(contingency_table)
>>>
>>> # Print the results
>>> print(f"Chi-squared value: {chi2}, p-value: {p}")
Chi-squared value: 0.0, p-value: 1.0
>>>


>>> import pandas as pd
>>>
>>> # CRO data for mobile device category
>>> data = {
...     'Ga:shoppingStage': ['add_to_cart', 'add_to_cart', 'add_to_cart', 'page_view', 'page_view', 'page_view', 'purchase', 'purchase', 'purchase', 'view_item', 'view_item', 'view_item'],
...     'Ga:deviceCategory': ['mobile', 'mobile', 'mobile', 'mobile', 'mobile', 'mobile', 'mobile', 'mobile', 'mobile', 'mobile', 'mobile', 'mobile'],
...     'Ga:month': [202402, 202403, 202404, 202402, 202403, 202404, 202402, 202403, 202404, 202402, 202403, 202404],
...     'Sum of Users': [518, 113, 64, 6242, 1073, 554, 287, 36, 21, 5140, 678, 248]
... }
>>>
>>> df = pd.DataFrame(data)
>>>
>>> # Convert 'Ga:month' to datetime and extract month number
>>> df['Ga:month'] = pd.to_datetime(df['Ga:month'], format='%Y%m')
>>> df['Month'] = df['Ga:month'].dt.month
>>>
>>> # Encode 'Ga:shoppingStage' as numeric for decision tree analysis
>>> stage_map = {'add_to_cart': 0, 'page_view': 1, 'purchase': 2, 'view_item': 3}
>>> df['Stage_Code'] = df['Ga:shoppingStage'].map(stage_map)
>>>
>>> print(df)
   Ga:shoppingStage Ga:deviceCategory   Ga:month  Sum of Users  Month  Stage_Code
0       add_to_cart            mobile 2024-02-01           518      2           0
1       add_to_cart            mobile 2024-03-01           113      3           0
2       add_to_cart            mobile 2024-04-01            64      4           0
3         page_view            mobile 2024-02-01          6242      2           1
4         page_view            mobile 2024-03-01          1073      3           1
5         page_view            mobile 2024-04-01           554      4           1
6          purchase            mobile 2024-02-01           287      2           2
7          purchase            mobile 2024-03-01            36      3           2
8          purchase            mobile 2024-04-01            21      4           2
9         view_item            mobile 2024-02-01          5140      2           3
10        view_item            mobile 2024-03-01           678      3           3
11        view_item            mobile 2024-04-01           248      4           3
>>>
>>>
>>> from sklearn.tree import DecisionTreeClassifier, plot_tree
>>> import matplotlib.pyplot as plt
>>>
>>> # Create decision tree model
>>> X = df[['Month']]
>>> y = df['Stage_Code']
>>> model = DecisionTreeClassifier()
>>> model.fit(X, y)
DecisionTreeClassifier()
>>>
>>> # Visualize decision tree
>>> plt.figure(figsize=(12, 6))
<Figure size 1200x600 with 0 Axes>
>>> plot_tree(model, feature_names=['Month'], class_names=['add_to_cart', 'page_view', 'purchase', 'view_item'], filled=True)
[Text(0.4, 0.8333333333333334, 'Month <= 2.5\ngini = 0.75\nsamples = 12\nvalue = [3, 3, 3, 3]\nclass = add_to_cart'), Text(0.2, 0.5, 'gini = 0.75\nsamples = 4\nvalue = [1, 1, 1, 1]\nclass = add_to_cart'), Text(0.6, 0.5, 'Month <= 3.5\ngini = 0.75\nsamples = 8\nvalue = [2, 2, 2, 2]\nclass = add_to_cart'), Text(0.4, 0.16666666666666666, 'gini = 0.75\nsamples = 4\nvalue = [1, 1, 1, 1]\nclass = add_to_cart'), Text(0.8, 0.16666666666666666, 'gini = 0.75\nsamples = 4\nvalue = [1, 1, 1, 1]\nclass = add_to_cart')]
>>> plt.show()
>>>



>>> # Create decision tree model
>>> X = df[['Month']]
>>> y = df['Stage_Code']
>>> model = DecisionTreeClassifier()
>>> model.fit(X, y)
DecisionTreeClassifier()
>>>
>>> # Visualize decision tree
>>> plt.figure(figsize=(12, 6))
<Figure size 1200x600 with 0 Axes>
>>> plot_tree(model, feature_names=['Month'], class_names=['add_to_cart', 'page_view', 'purchase', 'view_item'], filled=True)
[Text(0.4, 0.8333333333333334, 'Month <= 2.5\ngini = 0.75\nsamples = 12\nvalue = [3, 3, 3, 3]\nclass = add_to_cart'), Text(0.2, 0.5, 'gini = 0.75\nsamples = 4\nvalue = [1, 1, 1, 1]\nclass = add_to_cart'), Text(0.6, 0.5, 'Month <= 3.5\ngini = 0.75\nsamples = 8\nvalue = [2, 2, 2, 2]\nclass = add_to_cart'), Text(0.4, 0.16666666666666666, 'gini = 0.75\nsamples = 4\nvalue = [1, 1, 1, 1]\nclass = add_to_cart'), Text(0.8, 0.16666666666666666, 'gini = 0.75\nsamples = 4\nvalue = [1, 1, 1, 1]\nclass = add_to_cart')]
>>> plt.show()
>>>
>>> X = df[['Month', 'Sum of Users']]
>>> y = df['Stage_Code']
>>> model = DecisionTreeClassifier()
>>> model.fit(X, y)
DecisionTreeClassifier()
>>>
>>> plt.figure(figsize=(12, 6))
<Figure size 1200x600 with 0 Axes>
>>> plot_tree(model, feature_names=['Month', 'Sum of Users'], class_names=['add_to_cart', 'page_view', 'purchase', 'view_item'], filled=True)
[Text(0.36363636363636365, 0.9166666666666666, 'Sum of Users <= 536.0\ngini = 0.75\nsamples = 12\nvalue = [3, 3, 3, 3]\nclass = add_to_cart'), Text(0.18181818181818182, 0.75, 'Sum of Users <= 50.0\ngini = 0.612\nsamples = 7\nvalue = [3, 0, 3, 1]\nclass = add_to_cart'), Text(0.09090909090909091, 0.5833333333333334, 'gini = 0.0\nsamples = 2\nvalue = [0, 0, 2, 0]\nclass = purchase'), Text(0.2727272727272727, 0.5833333333333334, 'Sum of Users <= 180.5\ngini = 0.56\nsamples = 5\nvalue = [3, 0, 1, 1]\nclass = add_to_cart'), Text(0.18181818181818182, 0.4166666666666667, 'gini = 0.0\nsamples = 2\nvalue = [2, 0, 0, 0]\nclass = add_to_cart'), Text(0.36363636363636365, 0.4166666666666667, 'Sum of Users <= 267.5\ngini = 0.667\nsamples = 3\nvalue = [1, 0, 1, 1]\nclass = add_to_cart'), Text(0.2727272727272727, 0.25, 'gini = 0.0\nsamples = 1\nvalue = [0, 0, 0, 1]\nclass = view_item'), Text(0.45454545454545453, 0.25, 'Sum of Users <= 402.5\ngini = 0.5\nsamples = 2\nvalue = [1, 0, 1, 0]\nclass = add_to_cart'), Text(0.36363636363636365, 0.08333333333333333, 'gini = 0.0\nsamples = 1\nvalue = [0, 0, 1, 0]\nclass = purchase'), Text(0.5454545454545454, 0.08333333333333333, 'gini = 0.0\nsamples = 1\nvalue = [1, 0, 0, 0]\nclass = add_to_cart'), Text(0.5454545454545454, 0.75, 'Sum of Users <= 616.0\ngini = 0.48\nsamples = 5\nvalue = [0, 3, 0, 2]\nclass = page_view'), Text(0.45454545454545453, 0.5833333333333334, 'gini = 0.0\nsamples = 1\nvalue = [0, 1, 0, 0]\nclass = page_view'), Text(0.6363636363636364, 0.5833333333333334, 'Sum of Users <= 875.5\ngini = 0.5\nsamples = 4\nvalue = [0, 2, 0, 2]\nclass = page_view'), Text(0.5454545454545454, 0.4166666666666667, 'gini = 0.0\nsamples = 1\nvalue = [0, 0, 0, 1]\nclass = view_item'), Text(0.7272727272727273, 0.4166666666666667, 'Sum of Users <= 3106.5\ngini = 0.444\nsamples = 3\nvalue = [0, 2, 0, 1]\nclass = page_view'), Text(0.6363636363636364, 0.25, 'gini = 0.0\nsamples = 1\nvalue = [0, 1, 0, 0]\nclass = page_view'), Text(0.8181818181818182, 0.25, 'Sum of Users <= 5691.0\ngini = 0.5\nsamples = 2\nvalue = [0, 1, 0, 1]\nclass = page_view'), Text(0.7272727272727273, 0.08333333333333333, 'gini = 0.0\nsamples = 1\nvalue = [0, 0, 0, 1]\nclass = view_item'), Text(0.9090909090909091, 0.08333333333333333, 'gini = 0.0\nsamples = 1\nvalue = [0, 1, 0, 0]\nclass = page_view')]
>>> plt.show()
>>>
>>> y_pred = model.predict(X)
>>> accuracy = accuracy_score(y, y_pred)
>>> print('Decision Tree Accuracy:', accuracy)
Decision Tree Accuracy: 1.0
>>>
