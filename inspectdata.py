Python 3.12.3 (tags/v3.12.3:f6650f9, Apr  9 2024, 14:05:25) [MSC v.1938 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> import pandas as pd
... 
>>> file_paths = [
...     'C:\\Users\\t14\\OneDrive\\Documents\\CRO Data.xlsx',
...     'C:\\Users\\t14\\OneDrive\\Documents\\Google Ads Data.xlsx',
...     'C:\\Users\\t14\\OneDrive\\Documents\\Traffic Data.xlsx',
...     'C:\\Users\\t14\\OneDrive\\Documents\\Customer Segmentation.xlsx',
...     'C:\\Users\\t14\\OneDrive\\Documents\\Facebook Ads Data.xlsx',
...     'C:\\Users\\t14\\OneDrive\\Documents\\Sales Data.xlsx'
... ]
>>> dfs = {}
>>> for file_path in file_paths:
...     df = pd.read_excel(file_path)
...     dfs[file_path] = {
...         'duplicates': df.duplicated().sum(),
...         'missing_values': df.isnull().sum(),
...         'shape': df.shape
...     }
... 
...     
>>> for key, value in dfs.items():
...     print(f'File: {key}')
...     print('Duplicates:', value['duplicates'])
...     print('Missing Values:', value['missing_values'].to_dict())
...     print('Shape:', value['shape'])
... 
...     
File: C:\Users\t14\OneDrive\Documents\CRO Data.xlsx
Duplicates: 0
Missing Values: {'Ga4 Property': 0, 'Ga:shoppingStage': 0, 'Ga:deviceCategory': 0, 'Ga:month': 0, 'Sum of Users': 0, 'Sum of Transactions': 550, 'Sum of TransactionRevenue': 545, 'Unnamed: 7': 545, 'Unnamed: 8': 545, 'Unnamed: 9': 545, 'Unnamed: 10': 545}
Shape: (550, 11)
File: C:\Users\t14\OneDrive\Documents\Google Ads Data.xlsx
Duplicates: 0
Missing Values: {'Month': 0, 'Clicks': 0, 'Impr.': 0, 'CTR': 0, 'Currency code': 0, 'Avg. CPC': 0, 'Cost': 0, 'Impr. (Abs. Top) %': 0, 'Impr. (Top) %': 0, 'Conversions': 0, 'View-through conv.': 0, 'Cost / conv.': 0, 'Conv. rate': 0, 'All conv.': 0, 'Conv. value': 0}
Shape: (16, 15)
File: C:\Users\t14\OneDrive\Documents\Traffic Data.xlsx
Duplicates: 17
Missing Values: {'Customer Google Analytics → Property Name': 0, 'Ga4 Property': 0, 'Sum of Users': 0, 'Sum of Transactions': 0, 'Sum of TransactionRevenue': 0, 'Ga:sourceMedium': 0, 'Ga:month': 0}
Shape: (489, 7)
File: C:\Users\t14\OneDrive\Documents\Customer Segmentation.xlsx
Duplicates: 0
Missing Values: {'Customer UUID': 0, 'Platform Created At': 0, 'Gross sales': 0, 'Distinct values of Order UUID': 0}
Shape: (795, 4)
File: C:\Users\t14\OneDrive\Documents\Facebook Ads Data.xlsx
Duplicates: 0
Missing Values: {'Adset Name': 0, 'Date Start': 0, 'Spend': 9, 'Adset Actions → Purchase': 474, 'Adset Action Values → Omni Purchase': 432, 'Adset Actions → Link Click': 18, 'Ad Account → Name': 0}
Shape: (725, 7)
File: C:\Users\t14\OneDrive\Documents\Sales Data.xlsx
Duplicates: 0
Missing Values: {'Platform Created At': 0, 'Connection → Main Currency': 0, 'Gross sales': 0, 'Distinct values of Order UUID': 0, 'Refunds': 0}
Shape: (12, 5)
