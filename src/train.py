from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)
from sklearn.tree import DecisionTreeRegressor


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "food_delivery_featured.csv"
)

MODELS_DIRECTORY = PROJECT_ROOT / "models"

RESULTS_PATH = (
    MODELS_DIRECTORY
    / "model_comparison.csv"
)

MODEL_PATH = (
    MODELS_DIRECTORY
    / "best_baseline_pipeline.joblib"
)

TARGET_COLUMN = "Time_taken(min)"


def load_data(data_path=DATA_PATH):
    """Load the featured dataset."""

    dataframe = pd.read_csv(data_path)

    print("Dataset loaded:", data_path)
    print("Dataset shape:", dataframe.shape)

    return dataframe


def split_data(dataframe):
    """
    Separate the explanatory variables from the target
    and split the dataset into 80% training and 20% testing.
    """

    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' was not found."
        )

    X = dataframe.drop(
        columns=[TARGET_COLUMN]
    )

    y = dataframe[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))

    return X_train, X_test, y_train, y_test


def get_column_types(X_train):
    """Identify numerical and categorical columns."""

    numerical_columns = (
        X_train
        .select_dtypes(include="number")
        .columns
        .tolist()
    )

    categorical_columns = (
        X_train
        .select_dtypes(exclude="number")
        .columns
        .tolist()
    )

    return numerical_columns, categorical_columns


def create_preprocessor(X_train):
    """Create preprocessing for each type of feature."""

    numerical_columns, categorical_columns = (
        get_column_types(X_train)
    )

    numerical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        ),
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        ),
    ])

    preprocessor = ColumnTransformer([
        (
            "numerical",
            numerical_pipeline,
            numerical_columns
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_columns
        ),
    ])

    return preprocessor


def create_models():
    """Create four regression models."""

    models = {
        "Linear Regression": LinearRegression(),

        "Ridge Regression": Ridge(
            alpha=1.0
        ),

        "Decision Tree": DecisionTreeRegressor(
            random_state=42
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=150,
            random_state=42,
            n_jobs=-1
        ),
    }

    return models


def calculate_metrics(y_true, predictions):
    """Calculate regression metrics."""

    mse = mean_squared_error(
        y_true,
        predictions
    )

    metrics = {
        "MAE": mean_absolute_error(
            y_true,
            predictions
        ),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(
            y_true,
            predictions
        ),
    }

    return metrics


def train_and_evaluate(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor,
):
    """Train and evaluate all regression models."""

    models = create_models()

    results = []
    trained_pipelines = {}

    for model_name, model in models.items():
        print(f"Training {model_name}...")

        pipeline = Pipeline([
            (
                "preprocessor",
                clone(preprocessor)
            ),
            (
                "model",
                model
            ),
        ])

        pipeline.fit(
            X_train,
            y_train
        )

        train_predictions = pipeline.predict(
            X_train
        )

        test_predictions = pipeline.predict(
            X_test
        )

        train_metrics = calculate_metrics(
            y_train,
            train_predictions
        )

        test_metrics = calculate_metrics(
            y_test,
            test_predictions
        )

        results.append({
            "Model": model_name,

            "Train_MAE": train_metrics["MAE"],
            "Test_MAE": test_metrics["MAE"],

            "Train_MSE": train_metrics["MSE"],
            "Test_MSE": test_metrics["MSE"],

            "Train_RMSE": train_metrics["RMSE"],
            "Test_RMSE": test_metrics["RMSE"],

            "Train_R2": train_metrics["R2"],
            "Test_R2": test_metrics["R2"],
        })

        trained_pipelines[model_name] = pipeline

    results_dataframe = pd.DataFrame(
        results
    )

    results_dataframe = (
        results_dataframe
        .sort_values(by="Test_RMSE")
        .reset_index(drop=True)
    )

    return results_dataframe, trained_pipelines


def save_results(
    results_dataframe,
    trained_pipelines,
):
    """Save model results and the best baseline pipeline."""

    MODELS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    best_model_name = results_dataframe.loc[
        0,
        "Model"
    ]

    best_pipeline = trained_pipelines[
        best_model_name
    ]

    results_dataframe.to_csv(
        RESULTS_PATH,
        index=False
    )

    joblib.dump(
        best_pipeline,
        MODEL_PATH
    )

    print("Best baseline model:", best_model_name)
    print("Results saved to:", RESULTS_PATH)
    print("Pipeline saved to:", MODEL_PATH)

    return best_model_name, best_pipeline


def main():
    dataframe = load_data()

    X_train, X_test, y_train, y_test = split_data(
        dataframe
    )

    preprocessor = create_preprocessor(
        X_train
    )

    results, trained_pipelines = train_and_evaluate(
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    )

    print(
        results
        .round(3)
        .to_string(index=False)
    )

    save_results(
        results,
        trained_pipelines
    )


if __name__ == "__main__":
    main()