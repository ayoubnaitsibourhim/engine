import pandas as pd

def generate_recommendations_unit_economics(file_path):
    # Load the Unit Economics data from the Excel file
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

# Example usage
# file_path_unit_economics = "path_to_unit_economics_data.xlsx"
# generate_recommendations_unit_economics(file_path_unit_economics)
