# recommendation_engine.py

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

def generate_recommendations_facebook(df):
    # Analyze the metrics and generate recommendations
    recommendations = []
    if df['CostPerClick'].mean() > 0.5:
        recommendations.append("Lower CPC prices")
    if df['Cost'].sum() < 10000:
        recommendations.append("Scale ad spend")
    if df['ClickToPurchaseConversion'].mean() < 0.05:
        recommendations.append("Improve conversion rate")

    # Output the recommendations
    return recommendations

def train_models(df):
    # Assuming the Excel file has columns: Cost, Purchases, Revenue, Clicks, CPC, Click to purchase conversion, CAC, ROAS, AOV
    X = df[['Cost', 'Purchases', 'Revenue', 'Clicks']]
    y_ad_spend = df['Cost']
    y_cpc = df['Cost per Click']
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

    return lr_ad_spend_model, rf_ad_spend_model, lr_cpc_model, rf_cpc_model, lr_conversion_model, rf_conversion_model

def predict_and_generate_recommendations(df, models):
    lr_ad_spend_model, rf_ad_spend_model, lr_cpc_model, rf_cpc_model, lr_conversion_model, rf_conversion_model = models
    # Assume new_data is the new data received from the user
    new_data = df.copy()

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
    recommendations = []
    return recommendations

# Example usage
# Load the Excel file
file_path = r"C:\Users\t14\Downloads\fbdata.xlsx"
df = pd.read_excel(file_path)

# Generate recommendations for Facebook Ads
facebook_recommendations = generate_recommendations_facebook(df)

# Train models for Facebook Ads
models = train_models(df)

# Predict and generate recommendations based on the new data
new_recommendations = predict_and_generate_recommendations(df, models)

# Output the recommendations
print("\nRecommendations based on Facebook Ads data:")
for recommendation in facebook_recommendations:
    print(recommendation)

print("\nNew recommendations based on predictions:")
for recommendation in new_recommendations:
    print(recommendation)
