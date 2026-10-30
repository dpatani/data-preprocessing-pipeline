
from pathlib import Path
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent
INPUT_FILE = PROJECT_DIR / "data" / "Uncleaned_Global_Sales.csv"
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_FILE = OUTPUT_DIR / "Transformed_Global_Sales.csv"

# Define columns
NUMERICAL_FEATURES = ["Jan_Sales", "Feb_Sales", "Mar_Sales"]

CATEGORICAL_FEATURES = [
    "Product",
    "CustomerID",
    "OrderID",
    "ZipCode"
]


def main():

    # Step 1: Load dataset
    df = pd.read_csv(INPUT_FILE)

    print("\n--- ORIGINAL DATASET ---")
    print(df.head())

    print("\nDataset shape:", df.shape)
    print("\nMissing values before preprocessing:")
    print(df.isnull().sum())

    # Step 2: Numerical preprocessing
    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Step 3: Categorical preprocessing
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Step 4: Combine pipelines
    preprocessor = ColumnTransformer(transformers=[
        ("num", numerical_pipeline, NUMERICAL_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES)
    ])

    # Step 5: Apply preprocessing
    transformed_array = preprocessor.fit_transform(df)

    # Convert sparse matrix to dense array if required
    if hasattr(transformed_array, "toarray"):
        transformed_array = transformed_array.toarray()

    feature_names = preprocessor.get_feature_names_out()

    processed_df = pd.DataFrame(
        transformed_array,
        columns=feature_names,
        index=df.index
    )

    # Step 6: Save processed dataset
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    processed_df.to_csv(OUTPUT_FILE, index=False)

    # Step 7: Display results
    print("\n--- PREPROCESSING COMPLETED ---")
    print("Original dataset shape:", df.shape)
    print("Processed dataset shape:", processed_df.shape)

    print("\nMissing values after preprocessing:")
    print(processed_df.isnull().sum().sum())

    print("\nProcessed dataset preview:")
    print(processed_df.head())

    print("\nFile saved successfully:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
