Python 3.12.3 (tags/v3.12.3:f6650f9, Apr  9 2024, 14:05:25) [MSC v.1938 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> import pandas as pd
>>> from sklearn.cluster import KMeans
>>> from sklearn.cluster import DBSCAN
>>> from sklearn.model_selection import train_test_split
>>> from sklearn.ensemble import RandomForestClassifier
>>> from sklearn.tree import DecisionTreeClassifier
>>> from sklearn.metrics import accuracy_score
>>> file_path = "C:\\Users\\t14\\Documents\\Customer Segmentation.xlsx"
>>> df = pd.read_excel(file_path)
>>> kmeans = KMeans(n_clusters=3)
>>> kmeans.fit(df[['Gross sales', 'Distinct values of Order UUID']])
KMeans(n_clusters=3)
>>> df['KMeans_Cluster'] = kmeans.labels_
>>> dbscan = DBSCAN(eps=0.5, min_samples=5)
>>> dbscan.fit(df[['Gross sales', 'Distinct values of Order UUID']])
DBSCAN()
>>> df['DBSCAN_Cluster'] = dbscan.labels_
>>> X = df[['Gross sales', 'Distinct values of Order UUID']]
>>> y = df['KMeans_Cluster']
>>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
>>> dt_classifier = DecisionTreeClassifier()
>>> dt_classifier.fit(X_train, y_train)
DecisionTreeClassifier()
>>> dt_predictions = dt_classifier.predict(X_test)
>>> rf_classifier = RandomForestClassifier()
>>> rf_classifier.fit(X_train, y_train)
RandomForestClassifier()
>>> rf_predictions = rf_classifier.predict(X_test)
>>> dt_accuracy = accuracy_score(y_test, dt_predictions)
>>> rf_accuracy = accuracy_score(y_test, rf_predictions)
>>> print('Decision Tree Classifier Accuracy:', dt_accuracy)
Decision Tree Classifier Accuracy: 0.9937106918238994
>>> print('Random Forest Classifier Accuracy:', rf_accuracy)
Random Forest Classifier Accuracy: 0.9874213836477987
>>> >>> print(df[['Gross sales', 'Distinct values of Order UUID', 'KMeans_Cluster', 'DBSCAN_Cluster']].head())
   Gross sales  Distinct values of Order UUID  KMeans_Cluster  DBSCAN_Cluster
0        940.8                              2               1              -1
1        798.4                              2               1              -1
2        862.4                              2               1              -1
3        730.4                              2               0              -1
4       1101.6                              2               1              -1
>>>