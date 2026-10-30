# Global Sales Data Preprocessing Pipeline

A beginner-friendly, end-to-end data preprocessing project using Python, Pandas and Scikit-learn.

> **Dataset note:** The CSV included in `data/` is a small illustrative sample created for this repository so the project runs immediately. Replace it with your own dataset before presenting this as analysis of a real global sales dataset. Do not describe the sample data as real company data.

## Project structure

```text
Data-Preprocessing-Pipeline/
├── data/
│   └── Uncleaned_Global_Sales.csv
├── output/
│   └── .gitkeep
├── preprocessing_pipeline.py
├── requirements.txt
├── README.md
└── .gitignore
```

## What the project does

1. Loads a CSV file and inspects its dimensions, data types and missing values.
2. Separates numerical sales columns from categorical columns.
3. Fills missing numerical values using the median.
4. Fills missing categorical values using the most frequent value.
5. Standardises numerical features using `StandardScaler`.
6. Converts categorical features into indicator columns using `OneHotEncoder`.
7. Combines the transformations with `ColumnTransformer` and `Pipeline`.
8. Saves the transformed dataset as a CSV file.

## Dataset columns

| Column | Description | Processing |
|---|---|---|
| Product | Product category | Imputation + one-hot encoding |
| CustomerID | Customer identifier | Imputation + one-hot encoding |
| OrderID | Order identifier | Imputation + one-hot encoding |
| ZipCode | Postal code | Imputation + one-hot encoding |
| Jan_Sales | January sales | Median imputation + scaling |
| Feb_Sales | February sales | Median imputation + scaling |
| Mar_Sales | March sales | Median imputation + scaling |

IDs and postal codes are treated as categorical values, not quantities.

## Run locally

### 1. Install Python

Install Python 3.10 or newer.

### 2. Open a terminal in this project folder

Create and activate a virtual environment (recommended):

**Windows**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the pipeline

```bash
python preprocessing_pipeline.py
```

The processed file will be saved to:

`output/Transformed_Global_Sales.csv`

## Replace the sample dataset

Replace `data/Uncleaned_Global_Sales.csv` with your own CSV, keeping the expected column names, or edit the feature lists near the top of `preprocessing_pipeline.py` to match your dataset.

## Important machine-learning note

This script demonstrates preprocessing a complete dataset. In a predictive machine-learning project, split the data into training and test sets first, fit the preprocessing pipeline only on the training data, and use `transform()` on the test data. This prevents information from the test set leaking into training.

## Skills demonstrated

- Data inspection with Pandas
- Missing-value handling
- Numerical feature scaling
- Categorical feature encoding
- Reusable Scikit-learn pipelines
- Column-wise transformations
- CSV input/output
