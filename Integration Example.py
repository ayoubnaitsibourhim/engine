Python 3.12.3 (tags/v3.12.3:f6650f9, Apr  9 2024, 14:05:25) [MSC v.1938 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> # Assuming you have the purchase frequency values calculated from Customer Segmentation
>>> purchase_frequency = {
...     'Segment A': 1.2,
...     'Segment B': 1.5,
...     'Segment C': 1.1
... }
>>>
>>> # Pass the purchase frequency values to Unit Economics
>>> # For simplicity, I'll just print the values here
>>> print("Purchase Frequency Values:")
Purchase Frequency Values:
>>> for segment, frequency in purchase_frequency.items():
...     print(f"{segment}: {frequency}")
...
Segment A: 1.2
Segment B: 1.5
Segment C: 1.1
>>>
>>>
>>> # Receive the purchase frequency values from Customer Segmentation
>>> # For demonstration, I'll use the same purchase frequency values calculated above
>>> received_purchase_frequency = purchase_frequency
>>>
>>> # Use the purchase frequency values to calculate CLV and CAC for each customer segment
>>> # For demonstration, I'll just print the values here
>>> print("\nCalculating CLV and CAC based on Purchase Frequency:")

Calculating CLV and CAC based on Purchase Frequency:
>>> for segment, frequency in received_purchase_frequency.items():
...     clv = calculate_clv(frequency)  # Assuming you have a function to calculate CLV
...     cac = calculate_cac(clv)  # Assuming you have a function to calculate CAC
...     print(f"{segment}: CLV={clv}, CAC={cac}")
...
>>>
>>>
>>> def calculate_clv(purchase_frequency):
...     # Example calculation for demonstration
...     return purchase_frequency * 100  # Assuming a simple calculation for CLV
...
>>> def calculate_cac(clv):
...     # Example calculation for demonstration
...     return clv * 0.2  # Assuming a simple calculation for CAC
...
>>> # Receive the purchase frequency values from Customer Segmentation
>>> # For demonstration, I'll use the same purchase frequency values calculated above
>>> received_purchase_frequency = purchase_frequency
>>>
>>> # Use the purchase frequency values to calculate CLV and CAC for each customer segment
>>> # For demonstration, I'll just print the values here
>>> print("\nCalculating CLV and CAC based on Purchase Frequency:")

Calculating CLV and CAC based on Purchase Frequency:
>>> for segment, frequency in received_purchase_frequency.items():
...     clv = calculate_clv(frequency)
...     cac = calculate_cac(clv)
...     print(f"{segment}: CLV={clv}, CAC={cac}")
...
Segment A: CLV=120.0, CAC=24.0
Segment B: CLV=150.0, CAC=30.0
Segment C: CLV=110.00000000000001, CAC=22.000000000000004
>>>
>>>
>>>
>>> # Receive the purchase frequency values from Customer Segmentation
>>> # For demonstration, I'll use the same purchase frequency values calculated above
>>> received_purchase_frequency = purchase_frequency
>>>
>>> # Use the purchase frequency values to calculate CLV and CAC for each customer segment
>>> # For demonstration, I'll just print the values here
>>> print("\nCalculating CLV and CAC based on Purchase Frequency:")

Calculating CLV and CAC based on Purchase Frequency:
>>> for segment, frequency in received_purchase_frequency.items():
...     clv = calculate_clv(frequency)  # Assuming you have a function to calculate CLV
...     cac = calculate_cac(clv)  # Assuming you have a function to calculate CAC
...     print(f"{segment}: CLV={clv}, CAC={cac}")
...
Segment A: CLV=120.0, CAC=24.0
Segment B: CLV=150.0, CAC=30.0
Segment C: CLV=110.00000000000001, CAC=22.000000000000004
>>>
>>>
>>>
>>> # Receive the CLV and CAC values for each customer segment from Unit Economics
>>> # For demonstration, I'll use some sample values
>>> clv_values = {
...     'Segment A': 200,
...     'Segment B': 180,
...     'Segment C': 150
... }
>>> cac_values = {
...     'Segment A': 50,
...     'Segment B': 40,
...     'Segment C': 45
... }
>>>
>>> # Use these values to optimize ad spend and audience targeting for Facebook Ads
>>> # For demonstration, I'll just print the values here
>>> print("\nOptimizing Facebook Ads:")

Optimizing Facebook Ads:


>>>
>>> import pandas as pd
>>>
>>> # Load the Facebook Ads data
>>> data = {
...     'Month': ['Apr-24', 'Mar-24', 'Feb-24', 'Jan-24', 'Dec-23', 'Nov-23', 'Oct-23', 'Sep-23', 'Aug-23', 'Jul-23', 'Jun-23', 'May-23'],
...     'Cost': [10483, 18818, 18499, 86363, 91806, 31185, 22323, 19758, 18335, 12373, 4712, 3919],
...     'Purchases': [40, 58, 33, 96, 194, 61, 20, 36, 34, 39, 18, 17],
...     'Revenue': [104319, 129358, 84193, 265885, 465640, 204880, 44720, 66739, 62734, 68115, 34842, 15123],
...     'Clicks': [4169, 5793, 2690, 20032, 22264, 8310, 5985, 5652, 7637, 7223, 2401, 1596],
...     'CostPerClick': [2.5, 3.2, 6.9, 4.3, 4.1, 3.8, 3.7, 3.5, 2.4, 1.7, 2, 2.5],
...     'ClickToPurchaseConversion': [0.01, 0.01, 0.012, 0.005, 0.009, 0.007, 0.003, 0.006, 0.004, 0.005, 0.007, 0.011],
...     'CAC': [262, 324, 561, 900, 473, 511, 1116, 549, 539, 317, 262, 231],
...     'ROAS': [10, 6.9, 4.6, 3.1, 5.1, 6.6, 2, 3.4, 3.4, 5.5, 7.4, 3.9],
...     'AOV': [2607.976, 2230.308966, 2551.312121, 2769.635417, 2400.208454, 3358.683607, 2236.0175, 1853.852778, 1845.122941, 1746.525641, 1935.666667, 889.5941176]
... }
>>>
>>> df = pd.DataFrame(data)
>>>
>>> # Analyze the metrics and generate recommendations
>>> recommendations = []
>>> if df['CostPerClick'].mean() > 0.5:
...     recommendations.append("Lower CPC prices")
... if df['Cost'].sum() < 10000:  # Example budget
  File "<stdin>", line 3
    if df['Cost'].sum() < 10000:  # Example budget
    ^^
SyntaxError: invalid syntax
>>>     recommendations.append("Scale ad spend")
  File "<stdin>", line 1
    recommendations.append("Scale ad spend")

>>> if df['ClickToPurchaseConversion'].mean() < 0.05:
...     recommendations.append("Improve conversion rate")
...
>>> # Output the recommendations
>>> print("Recommendations based on Facebook Ads data:")
Recommendations based on Facebook Ads data:
>>> for recommendation in recommendations:
...     print(recommendation)
...
Improve conversion rate
>>>
>>>
>>> recommendations = []
>>> if df['CostPerClick'].mean() > 0.5:
...     recommendations.append("Lower CPC prices")
...

>>> if df['Cost'].sum() < 10000:
...     recommendations.append("Scale ad spend")
...
>>> if df['ClickToPurchaseConversion'].mean() < 0.05:
...     recommendations.append("Improve conversion rate")
...
>>> print("Recommendations based on Facebook Ads data:")
Recommendations based on Facebook Ads data:
>>> for recommendation in recommendations:
...     print(recommendation)
...
Lower CPC prices
Improve conversion rate
>>>



#Google Ads Integration


>>> import pandas as pd
>>> data_google = {
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
>>>
>>> df_google = pd.DataFrame(data_google)
>>>
>>> recommendations_google = []
>>> if df_google['Avg. CPC'].mean() > 0.5:
...     recommendations_google.append("Lower CPC prices")
...
>>> if df_google['Cost'].sum() < 10000:  # Example budget
...     recommendations_google.append("Scale ad spend")
...
>>> if df_google['Conv. rate'].str.rstrip('%').astype(float).mean() < 0.05:
...     recommendations_google.append("Improve conversion rate")
...
>>> print("Recommendations based on Google Ads data:")
Recommendations based on Google Ads data:
>>> for recommendation in recommendations_google:
...     print(recommendation)
...
Lower CPC prices
>>>