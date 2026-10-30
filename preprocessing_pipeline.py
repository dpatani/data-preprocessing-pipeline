"""
Global Sales Data Preprocessing Pipeline
Technologies: Python, Pandas, NumPy, Scikit-learn

Run from the project root:
    python preprocessing_pipeline.py
"""

from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_DIR = Path(__file__).resolve().parent
INPUT_FILE = PROJECT_DIR / "data" / "Uncleaned_Global_Sales.csv"
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_FILE = OUTPUT_DIR / "Transformed_Global_Sales.csv"

NUMERICAL_FEATURES = ["Jan_Sales", "Feb_Sales", "Mar_Sales"]
CATEGORICAL_FEATURES = ["Product", "CustomerID", "OrderID", "ZipCode"]


def main():
    # 1. Load and inspect the source data.
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}\n"
            "Place Uncleaned_Global_Sales.csv inside the data folder."
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    print("\\n--- ORIGINAL DATASET ---")
    print(df.head())
    print("\\nDataset shape:", df.shape)
    print("\\nColumn data types:\\n", df.dtypes)
    print("\\nMissing values before preprocessing:\\n", df.isnull().sum())

    # 2. Define separate preprocessing workflows.
    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    # 3. Apply the correct workflow to each group of columns.
    preprocessor = ColumnTransformer(transformers=[
        ("num", numerical_pipeline, NUMERICAL_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    # 4. Fit transformations and create the processed feature matrix.
    transformed_array = preprocessor.fit_transform(df)
    feature_names = preprocessor.get_feature_names_out()

    processed_df = pd.DataFrame(
        transformed_array,
        columns=feature_names,
        index=df.index,
    )

    # 5. Save the processed data.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(OUTPUT_FILE, index=False)

    print("\\n--- PREPROCESSING SUMMARY ---")
    print("Original shape:", df.shape)
    print("Processed shape:", processed_df.shape)
    print("Missing values after preprocessing:", int(processed_df.isnull().sum().sum()))
    print("\\nProcessed data preview:\\n", processed_df.head())
    print(f"\\nSaved processed dataset to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
