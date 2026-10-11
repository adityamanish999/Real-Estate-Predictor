import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os


def save(name):
    plt.tight_layout()
    plt.savefig(f'plots/{name}.png', dpi=100)
    plt.close()


def perform_eda(file_path):
    print("Loading clean data for EDA...")
    df = pd.read_csv(file_path)
    os.makedirs('plots', exist_ok=True)
    sns.set_theme(style="whitegrid")

    print("Generating EDA plots...")

    # ---- Q1-5: Price & size analysis ----

    # Q1. Distribution of property prices
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Price_in_Lakhs'], bins=50, kde=True)
    plt.title('Distribution of Property Prices (Lakhs)')
    plt.xlabel('Price (Lakhs)')
    plt.ylabel('Frequency')
    save('price_distribution')

    # Q2. Distribution of property sizes
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Size_in_SqFt'], bins=50, kde=True)
    plt.title('Distribution of Property Sizes (SqFt)')
    plt.xlabel('Size (SqFt)')
    plt.ylabel('Frequency')
    save('size_distribution')

    # Q3. Price per SqFt by property type
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='Property_Type', y='Price_per_SqFt')
    plt.title('Price per SqFt by Property Type')
    plt.xticks(rotation=45)
    save('price_by_type')

    # Q4. Size vs price
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df.sample(5000, random_state=42), x='Size_in_SqFt', y='Price_in_Lakhs', alpha=0.4)
    plt.title('Property Size vs Price (Sampled)')
    plt.xlabel('Size (SqFt)')
    plt.ylabel('Price (Lakhs)')
    save('q04_size_vs_price')

    # Q5. Outliers in price per sq ft and size
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(y=df['Price_per_SqFt'], ax=axes[0])
    axes[0].set_title('Price per SqFt (Lakhs) - outliers')
    sns.boxplot(y=df['Size_in_SqFt'], ax=axes[1])
    axes[1].set_title('Size (SqFt) - outliers')
    save('q05_outliers')

    # ---- Q6-10: Location-based analysis ----

    # Q6. Average price per sq ft by state
    state_pps = df.groupby('State')['Price_per_SqFt'].mean().sort_values(ascending=False)
    plt.figure(figsize=(10, 8))
    sns.barplot(x=state_pps.values, y=state_pps.index, color='steelblue')
    plt.title('Average Price per SqFt by State (Lakhs)')
    plt.xlabel('Avg Price per SqFt (Lakhs)')
    save('q06_pps_by_state')

    # Q7. Average property price by city
    city_price = df.groupby('City')['Price_in_Lakhs'].mean().sort_values(ascending=False)
    plt.figure(figsize=(10, 12))
    sns.barplot(x=city_price.values, y=city_price.index, color='teal')
    plt.title('Average Property Price by City (Lakhs)')
    plt.xlabel('Avg Price (Lakhs)')
    save('q07_price_by_city')

    # Q8. Median age of properties by locality (top 15 oldest of 500)
    loc_age = df.groupby('Locality')['Age_of_Property'].median().sort_values(ascending=False).head(15)
    plt.figure(figsize=(10, 7))
    sns.barplot(x=loc_age.values, y=loc_age.index, color='indianred')
    plt.title('Median Property Age by Locality (15 oldest localities)')
    plt.xlabel('Median Age (Years)')
    save('q08_median_age_by_locality')

    # Q9. BHK distribution across cities
    bhk_city = pd.crosstab(df['City'], df['BHK'])
    plt.figure(figsize=(8, 12))
    sns.heatmap(bhk_city, cmap='YlGnBu', cbar_kws={'label': 'Properties'})
    plt.title('BHK Distribution Across Cities')
    save('q09_bhk_by_city')

    # Q10 / Top 5 most expensive localities
    top_localities = df.groupby('Locality')['Price_in_Lakhs'].mean().sort_values(ascending=False).head(5)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_localities.values, y=top_localities.index)
    plt.title('Top 5 Most Expensive Localities (Avg Price)')
    plt.xlabel('Average Price (Lakhs)')
    save('top_localities')

    # Q10. Price trend (by year built) for those 5 localities
    trend = (df[df['Locality'].isin(top_localities.index)]
             .groupby(['Year_Built', 'Locality'])['Price_in_Lakhs'].mean().reset_index())
    plt.figure(figsize=(11, 6))
    sns.lineplot(data=trend, x='Year_Built', y='Price_in_Lakhs', hue='Locality', marker='o')
    plt.title('Price Trend by Year Built - Top 5 Expensive Localities')
    plt.ylabel('Avg Price (Lakhs)')
    save('q10_top5_price_trend')

    # ---- Q11-15: Feature relationships ----

    # Q11. Correlation heatmap
    plt.figure(figsize=(12, 10))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c not in ['ID', 'Growth_Rate', 'Good_Investment', 'Future_Price_5Y']]
    sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Heatmap of Numeric Features')
    save('correlation_heatmap')

    # Q12. Nearby schools vs price per sq ft
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Nearby_Schools', y='Price_per_SqFt', errorbar=None, color='steelblue')
    plt.title('Nearby Schools vs Average Price per SqFt')
    plt.ylabel('Avg Price per SqFt (Lakhs)')
    save('q12_schools_vs_pps')

    # Q13. Nearby hospitals vs price per sq ft
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Nearby_Hospitals', y='Price_per_SqFt', errorbar=None, color='seagreen')
    plt.title('Nearby Hospitals vs Average Price per SqFt')
    plt.ylabel('Avg Price per SqFt (Lakhs)')
    save('q13_hospitals_vs_pps')

    # Q14. Price by furnished status
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Furnished_Status', y='Price_in_Lakhs')
    plt.title('Price by Furnished Status')
    plt.ylabel('Price (Lakhs)')
    save('q14_price_by_furnishing')

    # Q15. Price per sq ft by facing direction
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Facing', y='Price_per_SqFt', errorbar=None, color='darkorange')
    plt.title('Average Price per SqFt by Facing Direction')
    plt.ylabel('Avg Price per SqFt (Lakhs)')
    save('q15_pps_by_facing')

    # ---- Q16-20: Investment / amenities / ownership ----

    # Q16. Properties per owner type
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(data=df, x='Owner_Type', color='slateblue')
    ax.bar_label(ax.containers[0])
    plt.title('Properties per Owner Type')
    save('q16_owner_type')

    # Q17. Properties per availability status
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(data=df, x='Availability_Status', color='slategray')
    ax.bar_label(ax.containers[0])
    plt.title('Properties per Availability Status')
    save('q17_availability')

    # Q18. Parking space vs price
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x='Parking_Space', y='Price_in_Lakhs', errorbar=None, color='goldenrod')
    plt.title('Parking Space vs Average Price')
    plt.ylabel('Avg Price (Lakhs)')
    save('q18_parking_vs_price')

    # Q19. Amenities vs price per sq ft
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x='Amenity_Score', y='Price_per_SqFt', errorbar=None, color='mediumpurple')
    plt.title('Number of Amenities vs Average Price per SqFt')
    plt.ylabel('Avg Price per SqFt (Lakhs)')
    save('q19_amenities_vs_pps')

    # Q20. Public transport vs price per sq ft and investment potential
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    order = ['Low', 'Medium', 'High']
    sns.barplot(data=df, x='Public_Transport_Accessibility', y='Price_per_SqFt', order=order,
                errorbar=None, color='steelblue', ax=axes[0])
    axes[0].set_title('Public Transport vs Avg Price per SqFt')
    sns.barplot(data=df, x='Public_Transport_Accessibility', y='Good_Investment', order=order,
                errorbar=None, color='seagreen', ax=axes[1])
    axes[1].set_title('Public Transport vs Share of Good Investments')
    save('q20_transport')

    # Extra: price vs age
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df.sample(5000, random_state=42), x='Age_of_Property', y='Price_in_Lakhs', alpha=0.5)
    plt.title('Property Price vs Age (Sampled)')
    plt.xlabel('Age of Property (Years)')
    plt.ylabel('Price (Lakhs)')
    save('price_vs_age')

    print("EDA completed. Plots saved in 'plots' directory.")


if __name__ == "__main__":
    perform_eda('data/clean_data_for_eda.csv')
