import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os

def select_features(input_path, output_path):
    print(f"Loading engineered data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # 1. Drop non-numeric columns for correlation analysis
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Target and Features
    X = numeric_df.drop(['Survived', 'PassengerId'], axis=1)
    y = numeric_df['Survived']
    
    # 2. Correlation Analysis
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    print(f"Dropping highly correlated features: {to_drop}")
    X = X.drop(to_drop, axis=1)
    
    # 3. Feature Importance using Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    importances = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
    importances = importances.sort_values(by='Importance', ascending=False)
    
    print("Top Features by Importance:")
    print(importances.head(10))
    
    # Select top 15 features
    selected_features = importances.head(15)['Feature'].tolist()
    
    # 4. Save selected features list
    with open(output_path, "w") as f:
        f.write("\n".join(selected_features))
    
    print(f"Selected features list saved to {output_path}")
    return selected_features

if __name__ == "__main__":
    select_features("data/train_engineered.csv", "data/selected_features.txt")
