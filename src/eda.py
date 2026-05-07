import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def perform_eda(file_path):
    print("Loading clean data for EDA...")
    df = pd.read_csv(file_path)
    os.makedirs('plots', exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    print("Generating EDA plots...")

    # 1. Price Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Price_in_Lakhs'], bins=50, kde=True)
    plt.title('Distribution of Property Prices (Lakhs)')
    plt.xlabel('Price (Lakhs)')
    plt.ylabel('Frequency')
    plt.savefig('plots/price_distribution.png')
    plt.close()

    # 2. Size Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Size_in_SqFt'], bins=50, kde=True)
    plt.title('Distribution of Property Sizes (SqFt)')
    plt.xlabel('Size (SqFt)')
    plt.ylabel('Frequency')
    plt.savefig('plots/size_distribution.png')
    plt.close()

    # 3. Price per SqFt by Property Type
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='Property_Type', y='Price_per_SqFt')
    plt.title('Price per SqFt by Property Type')
    plt.xticks(rotation=45)
    plt.savefig('plots/price_by_type.png')
    plt.close()

    # 4. Correlation Heatmap
    plt.figure(figsize=(12, 10))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # Drop IDs and some columns for cleaner heatmap
    numeric_cols = [c for c in numeric_cols if c not in ['ID', 'Growth_Rate', 'Good_Investment', 'Future_Price_5Y']]
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Heatmap of Numeric Features')
    plt.savefig('plots/correlation_heatmap.png')
    plt.close()

    # 5. Top 5 Most Expensive Localities
    top_localities = df.groupby('Locality')['Price_in_Lakhs'].mean().sort_values(ascending=False).head(5)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_localities.values, y=top_localities.index)
    plt.title('Top 5 Most Expensive Localities (Avg Price)')
    plt.xlabel('Average Price (Lakhs)')
    plt.savefig('plots/top_localities.png')
    plt.close()

    # 6. Price vs Age of Property
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df.sample(5000, random_state=42), x='Age_of_Property', y='Price_in_Lakhs', alpha=0.5)
    plt.title('Property Price vs Age (Sampled)')
    plt.xlabel('Age of Property (Years)')
    plt.ylabel('Price (Lakhs)')
    plt.savefig('plots/price_vs_age.png')
    plt.close()
    
    print("EDA completed. Plots saved in 'plots' directory.")

if __name__ == "__main__":
    perform_eda('data/clean_data_for_eda.csv')
