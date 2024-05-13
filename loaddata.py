Python 3.12.3 (tags/v3.12.3:f6650f9, Apr  9 2024, 14:05:25) [MSC v.1938 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> import pandas as pd
>>> file_paths = [
...     "C:/Users/t14/OneDrive/Documents/CRO Data.xlsx",
...     "C:/Users/t14/OneDrive/Documents/Google Ads Data.xlsx",
...     "C:/Users/t14/OneDrive/Documents/Traffic Data.xlsx",
...     "C:/Users/t14/OneDrive/Documents/Customer Segmentation.xlsx",
...     "C:/Users/t14/OneDrive/Documents/Facebook Ads Data.xlsx",
...     "C:/Users/t14/OneDrive/Documents/Sales Data.xlsx",
...     "C:/Users/t14/OneDrive/Documents/Unit Economics.docx"
... ]
>>> dfs = []
>>> for file_path in file_paths:
...     try:
...         if file_path.endswith(".xlsx"):
...             dfs.append(pd.read_excel(file_path))
...         elif file_path.endswith(".docx"):
...             # Handle docx files differently if needed
...             pass
...         else:
...             print(f"Unsupported file format for: {file_path}")
...     except Exception as e:
...         print(f"Error loading {file_path}: {e}")
... 
...         
>>> for i, df in enumerate(dfs):
...     print(f"DataFrame {i}:")
...     
...     print(df.head())  # Print the first few rows of each dataframe
... 
...     
DataFrame 0:
           Ga4 Property Ga:shoppingStage  ... Unnamed: 9  Unnamed: 10
0  properties/211991868      add_to_cart  ...        NaN          NaN
1  properties/211991868      add_to_cart  ...        NaN          NaN
2  properties/211991868      add_to_cart  ...        NaN          NaN
3  properties/211991868      add_to_cart  ...        NaN          NaN
4  properties/211991868      add_to_cart  ...        NaN          NaN

[5 rows x 11 columns]
DataFrame 1:
       Month  Clicks  Impr.  ...  Conv. rate All conv.  Conv. value
0 2023-01-01    1564  15443  ...      0.0254     87.77     41026.08
1 2023-02-01    2019  14427  ...      0.0490     98.92    104279.44
2 2023-03-01    2713  17835  ...      0.0209     56.78     62696.71
3 2023-04-01    3093  22619  ...      0.0184     58.17     69320.88
4 2024-04-01     207   6650  ...      0.0048      2.00      1377.45

[5 rows x 15 columns]
DataFrame 2:
  Customer Google Analytics → Property Name  ... Ga:month
0                             Totes Luxe UK  ...   202404
1                             Totes Luxe UK  ...   202403
2                             Totes Luxe UK  ...   202402
3                             Totes Luxe UK  ...   202401
4                             Totes Luxe UK  ...   202312

[5 rows x 7 columns]
DataFrame 3:
                          Customer UUID  ... Distinct values of Order UUID
0  02315305-a2fa-4d98-ba53-3f6ebfd6d6f5  ...                             2
1  120de52f-d3ac-48ef-8740-e0df5bb9cac9  ...                             2
2  451dab15-5342-4bca-9797-9ae3c5065247  ...                             2
3  5f5f7356-fc9e-4bb1-9851-e7b52c1e5a0c  ...                             2
4  7169901d-6623-4f09-86e2-718361f49ab9  ...                             2

[5 rows x 4 columns]
DataFrame 4:
                                 Adset Name  ... Ad Account → Name
0  (DV) ASC Performance // Evergreen Ad Set  ...             LuxUK
1  (DV) ASC Performance // Evergreen Ad Set  ...             LuxUK
2  (DV) ASC Performance // Evergreen Ad Set  ...             LuxUK
3  (DV) ASC Performance // Evergreen Ad Set  ...             LuxUK
4  (DV) ASC Performance // Evergreen Ad Set  ...             LuxUK

[5 rows x 7 columns]
DataFrame 5:
  Platform Created At  ...  Refunds
0          2024-04-01  ...  1293.45
1          2024-03-01  ...  2873.15
2          2024-02-01  ...  2422.00
3          2024-01-01  ...  1616.00
4          2023-12-01  ...  2908.75

[5 rows x 5 columns]

my_numbers = [129, 831, 14]  # Remove the leading zeros

my_octal_numbers = [0o12, 0o77, 0o14]  # Octal integers with the 0o prefix




