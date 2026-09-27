from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from train import (
    load_data,
    split_data,
    create_preprocessor,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIRECTORY = PROJECT_ROOT / "models"

FINAL_PIPELINE_PATH = (
    MODELS_DIRECTORY
    / "final_pipeline.joblib"
)

FINAL_MODEL_PATH = (
    MODELS_DIRECTORY
    / "final_random_forest.joblib"
)

FINAL_PREPROCESSOR_PATH = (
    MODELS_DIRECTORY
    / "final_preprocessor.joblib"
)

METRICS_PATH = (
    MODELS_DIRECTORY
    / "final_metrics.csv"
)

SEARCH_RESULTS_PATH = (
    MODELS_DIRECTORY
    / "optimization_results.csv"
)


def create_random_search(preprocessor):
    """Create the Random Forest hyperparameter search."""

    pipeline = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            RandomForestRegressor(
                random_state=42,
                n_jobs=1
            )
        ),
    ])

    parameter_distributions = {
        "model__n_estimators": [
            100,
            150,
            200,
            300,
        ],
        "model__max_depth": [
            8,
            12,
            16,
            20,
            None,
        ],
        "model__min_samples_split": [
            2,
            5,
            10,
            20,
        ],
        "model__min_samples_leaf": [
            1,
            2,
            4,
            8,
        ],
        "model__max_features": [
            "sqrt",
            "log2",
            0.7,
            1.0,
        ],
        "model__bootstrap": [
            True,
        ],
    }

    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=parameter_distributions,
        n_iter=15,
        scoring="neg_root_mean_squared_error",
        cv=5,
        random_state=42,
        n_jobs=2,
        verbose=2,
        return_train_score=True,
    )

    return random_search


def calculate_adjusted_r2(
    r2,
    number_of_rows,
    number_of_features,
):
    """Calculate adjusted R²."""

    if number_of_rows <= number_of_features + 1:
        return np.nan

    adjusted_r2 = 1 - (
        (1 - r2)
        * (number_of_rows - 1)
        / (
            number_of_rows
            - number_of_features
            - 1
        )
    )

    return adjusted_r2


def calculate_metrics(
    pipeline,
    X,
    y,
):
    """Calculate the regression metrics."""

    predictions = pipeline.predict(X)

    mse = mean_squared_error(
        y,
        predictions
    )

    mae = mean_absolute_error(
        y,
        predictions
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y,
        predictions
    )

    transformed_data = (
        pipeline
        .named_steps["preprocessor"]
        .transform(X)
    )

    number_of_features = transformed_data.shape[1]

    adjusted_r2 = calculate_adjusted_r2(
        r2=r2,
        number_of_rows=len(y),
        number_of_features=number_of_features,
    )

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "Adjusted_R2": adjusted_r2,
    }


def evaluate_final_model(
    pipeline,
    X_train,
    X_test,
    y_train,
    y_test,
):
    """Compare training and testing performance."""

    train_metrics = calculate_metrics(
        pipeline,
        X_train,
        y_train
    )

    test_metrics = calculate_metrics(
        pipeline,
        X_test,
        y_test
    )

    metrics_dataframe = pd.DataFrame([
        {
            "Dataset": "Training",
            **train_metrics,
        },
        {
            "Dataset": "Testing",
            **test_metrics,
        },
    ])

    return metrics_dataframe


def save_final_artifacts(
    random_search,
    metrics_dataframe,
):
    """Save the pipeline, model, preprocessor and results."""

    MODELS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    final_pipeline = random_search.best_estimator_

    final_preprocessor = (
        final_pipeline
        .named_steps["preprocessor"]
    )

    final_model = (
        final_pipeline
        .named_steps["model"]
    )

    joblib.dump(
        final_pipeline,
        FINAL_PIPELINE_PATH
    )

    joblib.dump(
        final_preprocessor,
        FINAL_PREPROCESSOR_PATH
    )

    joblib.dump(
        final_model,
        FINAL_MODEL_PATH
    )

    metrics_dataframe.to_csv(
        METRICS_PATH,
        index=False
    )

    search_results = pd.DataFrame(
        random_search.cv_results_
    )

    search_results.to_csv(
        SEARCH_RESULTS_PATH,
        index=False
    )

    print("\nFiles saved:")

    print(
        "Final pipeline:",
        FINAL_PIPELINE_PATH
    )

    print(
        "Final model:",
        FINAL_MODEL_PATH
    )

    print(
        "Final preprocessor:",
        FINAL_PREPROCESSOR_PATH
    )

    print(
        "Final metrics:",
        METRICS_PATH
    )

    print(
        "Search results:",
        SEARCH_RESULTS_PATH
    )


def main():
    dataframe = load_data()

    X_train, X_test, y_train, y_test = split_data(
        dataframe
    )

    preprocessor = create_preprocessor(
        X_train
    )

    random_search = create_random_search(
        preprocessor
    )

    print("\nStarting RandomizedSearchCV...")

    random_search.fit(
        X_train,
        y_train
    )

    print("\nBest parameters:")

    for parameter, value in (
        random_search
        .best_params_
        .items()
    ):
        print(f"{parameter}: {value}")

    print(
        "\nBest cross-validation RMSE:",
        round(-random_search.best_score_, 3)
    )

    final_pipeline = random_search.best_estimator_

    metrics_dataframe = evaluate_final_model(
        final_pipeline,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    print("\nFinal metrics:")

    print(
        metrics_dataframe
        .round(3)
        .to_string(index=False)
    )

    save_final_artifacts(
        random_search,
        metrics_dataframe
    )


if __name__ == "__main__":
    main()