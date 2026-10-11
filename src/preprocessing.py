import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_data(file_path):
    print("Loading data...")
    df = pd.read_csv(file_path)
    return df

def data_quality_report(df):
    """Fill missing values and flag (but keep) outliers. Saves a per-column summary."""
    print("Checking data quality...")
    num_cols = df.select_dtypes(include=[np.number]).columns.drop('ID', errors='ignore')
    cat_cols = df.select_dtypes(exclude=[np.number]).columns

    report = pd.DataFrame({'missing': df.isna().sum()})

    # Impute: median for numeric columns, mode for categorical columns
    for col in num_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # Outliers use the IQR rule. They are reported, not removed, so training data is unchanged.
    outliers = {}
    for col in ['Price_in_Lakhs', 'Size_in_SqFt', 'Price_per_SqFt']:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        outliers[col] = int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum())
    report['iqr_outliers'] = pd.Series(outliers)

    os.makedirs('data', exist_ok=True)
    report.to_csv('data/data_quality_report.csv', index_label='column')
    print(f"Missing values: {int(report['missing'].sum())}. IQR outliers: {outliers}")
    return df


def feature_engineering(df):
    print("Feature Engineering...")
    # Drop duplicates
    n_before = len(df)
    df = df.drop_duplicates()
    print(f"Removed {n_before - len(df)} duplicate rows.")
    df = data_quality_report(df)
    
    # 1. Amenity Score: count number of amenities
    df['Amenity_Score'] = df['Amenities'].apply(lambda x: len(str(x).split(',')))
    
    # 2. Good Investment: Price_per_SqFt < Locality median Price_per_SqFt
    locality_medians = df.groupby('Locality')['Price_per_SqFt'].transform('median')
    # Multi-factor: price < median AND BHK >= 2
    df['Good_Investment'] = ((df['Price_per_SqFt'] < locality_medians) & (df['BHK'] >= 2)).astype(int)
    
    # 3. Future Price 5Y
    # Generate a growth rate between 6% and 12% based on a hash of the city to keep it consistent but varied
    np.random.seed(42)
    city_growth_rates = {city: np.random.uniform(0.06, 0.12) for city in df['City'].unique()}
    df['Growth_Rate'] = df['City'].map(city_growth_rates)
    
    # Add some premium for ready to move
    df.loc[df['Availability_Status'] == 'Ready_to_Move', 'Growth_Rate'] += 0.01
    
    df['Future_Price_5Y'] = df['Price_in_Lakhs'] * ((1 + df['Growth_Rate']) ** 5)
    
    # Save the cleaned data for EDA before encoding/scaling
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/clean_data_for_eda.csv', index=False)
    
    return df

def encode_and_scale(df):
    print("Encoding and Scaling...")
    # Drop columns not needed for modeling
    drop_cols = ['ID', 'Amenities', 'Growth_Rate']
    df_model = df.drop(columns=drop_cols)
    
    # Target Encoding for high cardinality categoricals (State, City, Locality) using mean Price_in_Lakhs
    for col in ['State', 'City', 'Locality']:
        mean_encoding = df_model.groupby(col)['Price_in_Lakhs'].mean()
        df_model[col + '_Encoded'] = df_model[col].map(mean_encoding)
        df_model.drop(columns=[col], inplace=True)
        # save the encoding dict for the streamlit app
        joblib.dump(mean_encoding.to_dict(), f'models/{col}_encoding.pkl')
        
    # One-Hot Encoding for low cardinality
    categorical_cols = ['Property_Type', 'Furnished_Status', 'Public_Transport_Accessibility', 
                       'Parking_Space', 'Security', 'Facing', 'Owner_Type', 'Availability_Status']
    
    df_model = pd.get_dummies(df_model, columns=categorical_cols, drop_first=True)
    
    # Ensure all boolean columns are integers
    for col in df_model.columns:
        if df_model[col].dtype == 'bool':
            df_model[col] = df_model[col].astype(int)
            
    # Save feature columns list for later use
    features = [c for c in df_model.columns if c not in ['Good_Investment', 'Future_Price_5Y']]
    os.makedirs('models', exist_ok=True)
    joblib.dump(features, 'models/feature_columns.pkl')
    
    # Separate features and targets
    X = df_model.drop(columns=['Good_Investment', 'Future_Price_5Y'])
    
    # Scale numerical features (excluding encoded categoricals and targets)
    num_cols = ['BHK', 'Size_in_SqFt', 'Price_in_Lakhs', 'Price_per_SqFt', 'Year_Built', 
                'Floor_No', 'Total_Floors', 'Age_of_Property', 'Nearby_Schools', 
                'Nearby_Hospitals', 'Amenity_Score']
                
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])
    
    # Save the scaler
    joblib.dump(scaler, 'models/scaler.pkl')
    
    df_model[num_cols] = X[num_cols]
    
    # Save final modeling data
    df_model.to_csv('data/model_ready_data.csv', index=False)
    print("Preprocessing completed!")
    
if __name__ == "__main__":
    df_raw = load_data('india_housing_prices.csv')
    df_engineered = feature_engineering(df_raw)
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    encode_and_scale(df_engineered)
