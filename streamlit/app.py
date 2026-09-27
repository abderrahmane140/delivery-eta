from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split


# ---------------------------------------
# Configuration and paths
# ---------------------------------------

st.set_page_config(
    page_title="DeliveryETA",
    page_icon="🚚",
    layout="wide",
)

plt.style.use("dark_background")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "final_pipeline.joblib"
)

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "food_delivery_featured.csv"
)

METRICS_PATH = (
    PROJECT_ROOT
    / "models"
    / "final_metrics.csv"
)

COMPARISON_PATH = (
    PROJECT_ROOT
    / "models"
    / "model_comparison.csv"
)

TARGET_COLUMN = "Time_taken(min)"


# ---------------------------------------
# Load files
# ---------------------------------------

@st.cache_resource
def load_model():
    """Load the trained pipeline."""

    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset():
    """Load the featured delivery dataset."""

    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_metrics():
    """Load the final evaluation metrics."""

    return pd.read_csv(METRICS_PATH)


@st.cache_data
def load_model_comparison():
    """Load baseline model results."""

    return pd.read_csv(COMPARISON_PATH)


try:
    model_pipeline = load_model()
    dataframe = load_dataset()
    metrics_dataframe = load_metrics()
    comparison_dataframe = load_model_comparison()

except FileNotFoundError as error:
    st.error(f"Required file not found: {error}")
    st.stop()


# Get the exact columns expected by the model
feature_columns = list(
    model_pipeline.feature_names_in_
)


# ---------------------------------------
# Header
# ---------------------------------------

st.title("🚚 DeliveryETA")

st.write(
    "Predict the total delivery time using delivery "
    "conditions and order information."
)


# ---------------------------------------
# Application tabs
# ---------------------------------------

prediction_tab, data_tab, performance_tab = st.tabs([
    "ETA Prediction",
    "Data Visualization",
    "Model Performance",
])


# =======================================
# Prediction
# =======================================

with prediction_tab:
    st.subheader("Delivery information")

    st.info(
        "Complete the form and click Predict Delivery Time."
    )

    input_values = {}

    with st.form("prediction_form"):
        left_column, right_column = st.columns(2)

        categorical_columns = (
            dataframe[feature_columns]
            .select_dtypes(exclude="number")
            .columns
            .tolist()
        )

        numerical_columns = (
            dataframe[feature_columns]
            .select_dtypes(include="number")
            .columns
            .tolist()
        )

        with left_column:
            for column in categorical_columns:
                available_values = sorted(
                    dataframe[column]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                input_values[column] = st.selectbox(
                    label=column.replace("_", " "),
                    options=available_values,
                )

        with right_column:
            for column in numerical_columns:
                column_values = dataframe[column].dropna()

                if column == "order_hour":
                    input_values[column] = st.slider(
                        "Order hour",
                        min_value=0,
                        max_value=23,
                        value=int(column_values.median()),
                    )

                elif column == "distance_km":
                    input_values[column] = st.number_input(
                        "Delivery distance (km)",
                        min_value=0.0,
                        max_value=float(
                            column_values.max()
                        ),
                        value=float(
                            column_values.median()
                        ),
                        step=0.1,
                    )

                elif column == "preparation_time_minutes":
                    input_values[column] = st.number_input(
                        "Preparation time (minutes)",
                        min_value=0.0,
                        max_value=float(
                            column_values.max()
                        ),
                        value=float(
                            column_values.median()
                        ),
                        step=1.0,
                    )

                else:
                    input_values[column] = st.number_input(
                        column.replace("_", " "),
                        value=float(
                            column_values.median()
                        ),
                    )

        predict_button = st.form_submit_button(
            "Predict Delivery Time",
            type="primary",
            use_container_width=True,
        )

    if predict_button:
        input_dataframe = pd.DataFrame(
            [input_values],
            columns=feature_columns,
        )

        prediction = model_pipeline.predict(
            input_dataframe
        )[0]

        st.success(
            f"Estimated delivery time: "
            f"{prediction:.1f} minutes"
        )

        st.metric(
            label="Predicted ETA",
            value=f"{prediction:.1f} min",
        )

        if prediction <= 25:
            st.info("Expected delivery: Fast")

        elif prediction <= 40:
            st.warning("Expected delivery: Normal")

        else:
            st.error("Possible risk of delay")


# =======================================
# Data visualization
# =======================================

with data_tab:
    st.subheader("Dataset overview")

    metric_1, metric_2, metric_3 = st.columns(3)

    metric_1.metric(
        "Number of deliveries",
        f"{len(dataframe):,}",
    )

    metric_2.metric(
        "Average delivery time",
        f"{dataframe[TARGET_COLUMN].mean():.1f} min",
    )

    metric_3.metric(
        "Median delivery time",
        f"{dataframe[TARGET_COLUMN].median():.1f} min",
    )

    st.subheader("Delivery-time distribution")

    figure, axis = plt.subplots(
        figsize=(10, 5),
        facecolor="black",
    )

    sns.histplot(
        data=dataframe,
        x=TARGET_COLUMN,
        bins=30,
        kde=True,
        color="steelblue",
        ax=axis,
    )

    axis.set_title("Distribution of Delivery Time")
    axis.set_xlabel("Delivery time in minutes")
    axis.set_ylabel("Number of deliveries")
    axis.grid(alpha=0.2)

    st.pyplot(figure)
    plt.close(figure)

    first_graph, second_graph = st.columns(2)

    with first_graph:
        st.subheader("Average time by traffic")

        traffic_average = (
            dataframe
            .groupby(
                "Road_traffic_density"
            )[TARGET_COLUMN]
            .mean()
            .sort_values()
        )

        st.bar_chart(traffic_average)

    with second_graph:
        st.subheader("Average time by weather")

        weather_average = (
            dataframe
            .groupby(
                "Weatherconditions"
            )[TARGET_COLUMN]
            .mean()
            .sort_values()
        )

        st.bar_chart(weather_average)

    st.subheader("Dataset sample")

    st.dataframe(
        dataframe.head(20),
        use_container_width=True,
    )


# =======================================
# Model performance
# =======================================

with performance_tab:
    st.subheader("Final model metrics")

    test_metrics = (
        metrics_dataframe[
            metrics_dataframe["Dataset"] == "Testing"
        ]
        .iloc[0]
    )

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)

    metric_1.metric(
        "MAE",
        f"{test_metrics['MAE']:.2f} min",
    )

    metric_2.metric(
        "RMSE",
        f"{test_metrics['RMSE']:.2f} min",
    )

    metric_3.metric(
        "R²",
        f"{test_metrics['R2']:.3f}",
    )

    metric_4.metric(
        "Adjusted R²",
        f"{test_metrics['Adjusted_R2']:.3f}",
    )

    st.write(
        f"The final model makes an average error of "
        f"approximately **{test_metrics['MAE']:.2f} minutes**."
    )

    st.subheader("Model comparison")

    comparison_chart = (
        comparison_dataframe[
            [
                "Model",
                "Test_RMSE",
            ]
        ]
        .set_index("Model")
    )

    st.bar_chart(comparison_chart)

    st.subheader("Training versus testing")

    train_test_metrics = (
        metrics_dataframe[
            [
                "Dataset",
                "MAE",
                "RMSE",
            ]
        ]
        .set_index("Dataset")
    )

    st.bar_chart(train_test_metrics)

    st.subheader("Actual versus predicted delivery time")

    X = dataframe[feature_columns]
    y = dataframe[TARGET_COLUMN]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    test_predictions = model_pipeline.predict(
        X_test
    )

    prediction_dataframe = pd.DataFrame({
        "Actual": y_test.to_numpy(),
        "Predicted": test_predictions,
    })

    # Use a sample to keep the graph readable
    graph_dataframe = prediction_dataframe.sample(
        n=min(2000, len(prediction_dataframe)),
        random_state=42,
    )

    figure, axis = plt.subplots(
        figsize=(7, 7),
        facecolor="black",
    )

    sns.scatterplot(
        data=graph_dataframe,
        x="Actual",
        y="Predicted",
        alpha=0.3,
        color="steelblue",
        ax=axis,
    )

    minimum_value = min(
        graph_dataframe["Actual"].min(),
        graph_dataframe["Predicted"].min(),
    )

    maximum_value = max(
        graph_dataframe["Actual"].max(),
        graph_dataframe["Predicted"].max(),
    )

    axis.plot(
        [minimum_value, maximum_value],
        [minimum_value, maximum_value],
        color="red",
        linestyle="--",
        label="Perfect prediction",
    )

    axis.set_title("Actual vs Predicted Delivery Time")
    axis.set_xlabel("Actual time in minutes")
    axis.set_ylabel("Predicted time in minutes")
    axis.legend()
    axis.grid(alpha=0.2)

    st.pyplot(figure)
    plt.close(figure)