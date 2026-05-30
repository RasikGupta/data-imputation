import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

def load_sample_data():
    """Loads a real-world sample dataset containing authentic missing values."""
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    # Filter to specific numerical columns for structural demonstration
    return df[['PassengerId', 'Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']]

def run_preprocessing_pipeline():
    print("=== Phase 1: Initializing Pipeline & Loading Dense Dataset ===")
    df = load_sample_data()
    
    # 1. Isolate and count initial missing records
    initial_missing = df.isnull().sum()
    print("\nInitial Missing Records Count per Feature:")
    print(initial_missing[initial_missing > 0])
    
    print("\n=== Phase 2: Executing Exploratory Data Analysis (EDA) ===")
    # Calculate pre-imputation descriptive metrics (Variance & Mean)
    pre_mean = df['Age'].mean()
    pre_var = df['Age'].var()
    print(f"Pre-Imputation 'Age' Feature Metrics -> Mean: {pre_mean:.2f} | Variance: {pre_var:.2f}")

    print("\n=== Phase 3: Applying Baseline & Algorithmic Imputation ===")
    # Copying data frames to prevent side-effects / mutate original arrays
    df_mean_imputed = df.copy()
    df_knn_imputed = df.copy()

    # Strategy A: Statistical Baseline Strategy (Mean Imputation)
    df_mean_imputed['Age'] = df_mean_imputed['Age'].fillna(pre_mean)
    print("✓ Statistical Baseline Imputation (Mean) applied successfully.")

    # Strategy B: Advanced Algorithmic Mechanics (K-Nearest Neighbors Imputer)
    # Initialize KNN Imputer with 5 nearest neighbors as mathematical bounds
    knn_imputer = KNNImputer(n_neighbors=5)
    knn_array = knn_imputer.fit_transform(df_knn_imputed)
    
    # Reconvert array back to structured Pandas DataFrame
    df_knn_imputed = pd.DataFrame(knn_array, columns=df.columns)
    print("✓ Advanced Algorithmic Imputation (KNN Imputer) executed successfully.")

    print("\n=== Phase 4: Measuring Post-Imputation Variance & Bias ===")
    # Evaluate changes in distribution to look for structural biases
    post_mean_stat = df_mean_imputed['Age'].mean()
    post_var_stat = df_mean_imputed['Age'].var()
    
    post_mean_knn = df_knn_imputed['Age'].mean()
    post_var_knn = df_knn_imputed['Age'].var()

    print(f"Mean Strategy Metrics -> Mean: {post_mean_stat:.2f} | Variance: {post_var_stat:.2f}")
    print(f"KNN Strategy Metrics  -> Mean: {post_mean_knn:.2f} | Variance: {post_var_knn:.2f}")
    print("\nPipeline execution complete. Downstream dataset configuration ready for Model Training.")

if __name__ == "__main__":
    run_preprocessing_pipeline()