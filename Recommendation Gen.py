#Facebook Ads

import pandas as pd

def generate_recommendations_facebook(file_path):
    # Load the Facebook Ads data from the Excel file
    file_path = r"C:\Users\t14\Documents\Facebook Ads Data.xlsx"
    df = pd.read_excel(file_path)
    
    # Analyze the metrics and generate recommendations
    recommendations = []
    if df['CostPerClick'].mean() > 0.5:
        recommendations.append("Lower CPC prices")
    if df['Cost'].sum() < 10000:
        recommendations.append("Scale ad spend")
    if df['ClickToPurchaseConversion'].mean() < 0.05:
        recommendations.append("Improve conversion rate")
    
    # Output the recommendations
    print("\nRecommendations based on Facebook Ads data:")
    for recommendation in recommendations:
        print(recommendation)

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Load the Excel file
file_path = "path_to_your_excel_file.xlsx"
df = pd.read_excel(file_path)

# Assuming the Excel file has columns: Cost, Purchases, Revenue, Clicks, CPC, Click to purchase conversion, CAC, ROAS, AOV
X = df[['Cost', 'Purchases', 'Revenue', 'Clicks']]
y_ad_spend = df['Cost']
y_cpc = df['CPC']
y_conversion = df['Click to purchase conversion']

# Train linear regression model for ad spend
lr_ad_spend_model = LinearRegression()
lr_ad_spend_model.fit(X, y_ad_spend)

# Train random forest regression model for ad spend
rf_ad_spend_model = RandomForestRegressor()
rf_ad_spend_model.fit(X, y_ad_spend)

# Train linear regression model for CPC
lr_cpc_model = LinearRegression()
lr_cpc_model.fit(X, y_cpc)

# Train random forest regression model for CPC
rf_cpc_model = RandomForestRegressor()
rf_cpc_model.fit(X, y_cpc)

# Train linear regression model for conversion rate
lr_conversion_model = LinearRegression()
lr_conversion_model.fit(X, y_conversion)

# Train random forest regression model for conversion rate
rf_conversion_model = RandomForestRegressor()
rf_conversion_model.fit(X, y_conversion)

# Assume new_data is the new data received from the user
new_data = pd.DataFrame({
    'Cost': [10000],
    'Purchases': [30],
    'Revenue': [80000],
    'Clicks': [3000]
})

# Predict ad spend
lr_ad_spend_prediction = lr_ad_spend_model.predict(new_data)
rf_ad_spend_prediction = rf_ad_spend_model.predict(new_data)

# Predict CPC
lr_cpc_prediction = lr_cpc_model.predict(new_data)
rf_cpc_prediction = rf_cpc_model.predict(new_data)

# Predict conversion rate
lr_conversion_prediction = lr_conversion_model.predict(new_data)
rf_conversion_prediction = rf_conversion_model.predict(new_data)

# Generate recommendations based on the predictions
# For example, you can scale ad spend if predicted ad spend is below a certain threshold, lower CPC prices if predicted CPC is higher than desired,
# and increase the conversion rate if predicted conversion rate is below a certain threshold



#Google Ads 


import pandas as pd

def generate_recommendations(df):
    recommendations = []
    if df['Avg. CPC'].mean() > 0.5:
        recommendations.append("Lower CPC prices")
    if df['Cost'].sum() < 10000:
        recommendations.append("Scale ad spend")
    if df['Conv. rate'].mean() < 0.05:
        recommendations.append("Improve conversion rate")
    return recommendations

# Load the Google Ads data from the Excel file
file_path = r"C:\Users\t14\Documents\Google Ads Data.xlsx"
df = pd.read_excel(file_path)

# Generate recommendations based on the data
recommendations = generate_recommendations(df)

# Output the recommendations
print("\nRecommendations based on Google Ads data:")
for recommendation in recommendations:
    print(recommendation)


import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Load the Excel file for Google Ads
file_path_google_ads = "path_to_your_google_ads_excel_file.xlsx"
df_google_ads = pd.read_excel(file_path_google_ads)

# Assuming the Excel file has columns: Conv. value, Clicks, Impr., CTR, Avg. CPC, Cost, Conversions, Cost / conv., Conv. rate
X_google_ads = df_google_ads[['Conv. value', 'Clicks', 'Impr.', 'CTR', 'Avg. CPC', 'Cost', 'Conversions', 'Cost / conv.', 'Conv. rate']]
y_ctr = df_google_ads['CTR']
y_cpc = df_google_ads['Avg. CPC']

# Train logistic regression model for CTR
logistic_ctr_model = LogisticRegression()
logistic_ctr_model.fit(X_google_ads, y_ctr)

# Train random forest regression model for CTR
rf_ctr_model = RandomForestRegressor()
rf_ctr_model.fit(X_google_ads, y_ctr)

# Train linear regression model for CPC
lr_cpc_model_google_ads = LinearRegression()
lr_cpc_model_google_ads.fit(X_google_ads, y_cpc)

# Train gradient boosting regression model for CPC
gb_cpc_model_google_ads = GradientBoostingRegressor()
gb_cpc_model_google_ads.fit(X_google_ads, y_cpc)

# Assume new_data_google_ads is the new data received from the user for Google Ads
new_data_google_ads = pd.DataFrame({
    'Conv. value': [1000],
    'Clicks': [200],
    'Impr.': [5000],
    'CTR': [0.04],
    'Avg. CPC': [5],
    'Cost': [1000],
    'Conversions': [10],
    'Cost / conv.': [100],
    'Conv. rate': [0.1]
})

# Predict CTR
logistic_ctr_prediction_google_ads = logistic_ctr_model.predict(new_data_google_ads)
rf_ctr_prediction_google_ads = rf_ctr_model.predict(new_data_google_ads)

# Predict CPC
lr_cpc_prediction_google_ads = lr_cpc_model_google_ads.predict(new_data_google_ads)
gb_cpc_prediction_google_ads = gb_cpc_model_google_ads.predict(new_data_google_ads)

# Generate recommendations based on the predictions
# For example, you can adjust ad bids based on the predicted CTR and CPC




#Unit Economics


import pandas as pd

def generate_recommendations_unit_economics(file_path):
    # Load the first sheet of the Excel file
    file_path = = r"C:\Users\t14\Documents\Unit Economics.xlsx"
    df = pd.read_excel(file_path)
    
    # Calculate CAC % of AOV
    df['CAC_percentage_AOV'] = df['CAC'] / df['AOV'] * 100
    
    # Analyze the metrics and generate recommendations
    recommendations = []
    if df['CAC_percentage_AOV'].mean() > 10:
        recommendations.append("Target a lower CAC % of AOV")
    else:
        recommendations.append("Optimal CAC % of AOV")
    
    # Output the recommendations
    print("\nRecommendations based on Unit Economics data:")
    for recommendation in recommendations:
        print(recommendation)


#CRO

import pandas as pd

def generate_recommendations_cro(file_path):
    # Load the Excel file
    df = pd.read_excel(file_path)
    
    # Filter data for each device category
    mobile_data = df[df['Ga:deviceCategory'] == 'mobile']
    desktop_data = df[df['Ga:deviceCategory'] == 'desktop']
    tablet_data = df[df['Ga:deviceCategory'] == 'tablet']
    
    # Analyze the data and generate recommendations
    recommendations = {}
    
    # Mobile device recommendations
    mobile_recommendations = []
    if mobile_data['Ga:shoppingStage'].str.contains('view item').any():
        mobile_recommendations.append("Optimize view item stage for mobile")
    if mobile_data['Ga:shoppingStage'].str.contains('add to cart').any():
        mobile_recommendations.append("Optimize add to cart stage for mobile")
    if mobile_data['Ga:shoppingStage'].str.contains('purchase').any():
        mobile_recommendations.append("Optimize purchase stage for mobile")
    recommendations['Mobile'] = mobile_recommendations
    
    # Desktop device recommendations
    desktop_recommendations = []
    if desktop_data['Ga:shoppingStage'].str.contains('view item').any():
        desktop_recommendations.append("Optimize view item stage for desktop")
    if desktop_data['Ga:shoppingStage'].str.contains('add to cart').any():
        desktop_recommendations.append("Optimize add to cart stage for desktop")
    if desktop_data['Ga:shoppingStage'].str.contains('purchase').any():
        desktop_recommendations.append("Optimize purchase stage for desktop")
    recommendations['Desktop'] = desktop_recommendations
    
    # Tablet device recommendations
    tablet_recommendations = []
    if tablet_data['Ga:shoppingStage'].str.contains('view item').any():
        tablet_recommendations.append("Optimize view item stage for tablet")
    if tablet_data['Ga:shoppingStage'].str.contains('add to cart').any():
        tablet_recommendations.append("Optimize add to cart stage for tablet")
    if tablet_data['Ga:shoppingStage'].str.contains('purchase').any():
        tablet_recommendations.append("Optimize purchase stage for tablet")
    recommendations['Tablet'] = tablet_recommendations
    
    # Output the recommendations
    print("\nRecommendations based on CRO data:")
    for device, device_recommendations in recommendations.items():
        print(f"{device} device:")
        for recommendation in device_recommendations:
            print(f"- {recommendation}")

#Customer Segmentation 

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def customer_segmentation(df):
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(df[['Gross sales', 'Distinct values of Order UUID']])
    df['Cluster'] = kmeans.labels_

    X = df[['Gross sales', 'Distinct values of Order UUID']]
    y = df['Cluster']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    dt_classifier = DecisionTreeClassifier()
    dt_classifier.fit(X_train, y_train)
    dt_predictions = dt_classifier.predict(X_test)

    rf_classifier = RandomForestClassifier()
    rf_classifier.fit(X_train, y_train)
    rf_predictions = rf_classifier.predict(X_test)

    dt_accuracy = accuracy_score(y_test, dt_predictions)
    rf_accuracy = accuracy_score(y_test, rf_predictions)

    print('Decision Tree Classifier Accuracy:', dt_accuracy)
    print('Random Forest Classifier Accuracy:', rf_accuracy)

    return df

# Example usage
# Assume df contains the user's data
result_df = customer_segmentation(df)
print(result_df.head())

