import pandas as pd
import numpy as np
import os

def clean_data(input_path, output_path):
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # 1. Missing Value Handling
    # Identify missing values
    missing_summary = df.isnull().sum()
    print("Missing values summary:")
    print(missing_summary[missing_summary > 0])
    
    # Age: Impute with median
    df['Age'] = df['Age'].fillna(df['Age'].median())
    
    # Embarked: Impute with mode
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # Cabin: Too many missing values, but let's create an indicator before dropping
    df['HasCabin'] = df['Cabin'].apply(lambda x: 0 if pd.isna(x) else 1)
    # We will extract Deck later in feature engineering, for now keep it or mark as 'Unknown'
    df['Cabin'] = df['Cabin'].fillna('Unknown')
    
    # 2. Outlier Handling
    # Detect outliers in Fare using IQR
    Q1 = df['Fare'].quantile(0.25)
    Q3 = df['Fare'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Cap outliers in Fare
    df['Fare'] = np.clip(df['Fare'], lower_bound, upper_bound)
    
    # 3. Data Consistency
    # Sex is already consistent (male/female), but let's ensure lowercase
    df['Sex'] = df['Sex'].str.lower()
    
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    
    # 4. Save cleaned data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")
    return df

if __name__ == "__main__":
    clean_data("data/train.csv", "data/train_cleaned.csv")
