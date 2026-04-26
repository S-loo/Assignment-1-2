import pandas as pd
import numpy as np
import os
import re

def engineer_features(input_path, output_path):
    print(f"Loading cleaned data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # 1. Create Derived Features
    # FamilySize
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # IsAlone
    df['IsAlone'] = df['FamilySize'].apply(lambda x: 1 if x == 1 else 0)
    
    # Title extraction from Name
    def extract_title(name):
        title_search = re.search(' ([A-Za-z]+)\.', name)
        if title_search:
            return title_search.group(1)
        return ""

    df['Title'] = df['Name'].apply(extract_title)
    # Group rare titles
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
    # Deck extraction from Cabin
    df['Deck'] = df['Cabin'].apply(lambda x: x[0] if x != 'Unknown' else 'U')
    
    # Age groups
    bins = [0, 12, 18, 60, 120]
    labels = ['Child', 'Teen', 'Adult', 'Senior']
    df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels)
    
    # Fare per person
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']
    
    # 2. Categorical Encoding
    # Nominal encoding using get_dummies
    df = pd.get_dummies(df, columns=['Sex', 'Embarked', 'Title', 'Deck', 'AgeGroup'], drop_first=True)
    
    # 3. Feature Transformations
    # Log transform Fare (handle zeros by adding 1)
    df['Fare_Log'] = np.log1p(df['Fare'])
    
    # 4. Save engineered data
    df.to_csv(output_path, index=False)
    print(f"Engineered features saved to {output_path}")
    return df

if __name__ == "__main__":
    engineer_features("data/train_cleaned.csv", "data/train_engineered.csv")
