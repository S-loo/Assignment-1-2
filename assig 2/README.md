# Titanic Dataset Analysis - Assignment 2

## Objective
Build a predictive model for Titanic survival by performing data cleaning, feature engineering, and feature selection.

## Project Structure
- `data/`: Contains raw and processed datasets.
- `notebooks/`: Exploratory Data Analysis and Feature Engineering walkthrough.
- `scripts/`: Modular Python scripts for data pipeline.
  - `data_cleaning.py`: Handles missing values, outliers, and consistency.
  - `feature_engineering.py`: Creates derived features and encoding.
  - `feature_selection.py`: Ranks features using Random Forest importance.

## Approach
1. **Data Cleaning**: 
   - Imputed `Age` with median and `Embarked` with mode.
   - Capped `Fare` outliers using IQR.
   - Created `HasCabin` indicator.
2. **Feature Engineering**:
   - Extracted `Title` and `Deck`.
   - Created `FamilySize` and `IsAlone`.
   - Binned `Age` into groups.
3. **Feature Selection**:
   - Used correlation matrix to remove redundant features.
   - Applied Random Forest importance to select top predictors.

## Instructions
1. Install dependencies: `pip install -r requirements.txt`
2. Run data cleaning: `python scripts/data_cleaning.py`
3. Run feature engineering: `python scripts/feature_engineering.py`
4. Run feature selection: `python scripts/feature_selection.py`
5. Explore the notebook in `notebooks/Titanic_Feature_Engineering.ipynb`.

## Key Findings
- **Gender** and **Pclass** remain the strongest predictors of survival.
- **Title** (e.g., 'Mr' vs 'Mrs') provides significant information.
- **FamilySize** shows that small families had better survival rates than individuals or large families.
