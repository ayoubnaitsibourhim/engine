import pandas as pd

def generate_recommendations_google_ads(file_path):
    # Load the Google Ads data from the Excel file
    df = pd.read_excel(file_path)

    # Analyze the metrics and generate recommendations
    recommendations = []
    if df['Avg. CPC'].mean() > 0.5:
        recommendations.append("Lower CPC prices")
    if df['Cost'].sum() < 10000:
        recommendations.append("Scale ad spend")
    if df['Conv. rate'].mean() < 0.05:
        recommendations.append("Improve conversion rate")
    
    # Output the recommendations
    print("\nRecommendations based on Google Ads data:")
    for recommendation in recommendations:
        print(recommendation)

# Example usage
# file_path_google_ads = "path_to_google_ads_data.xlsx"
# generate_recommendations_google_ads(file_path_google_ads)
