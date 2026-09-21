# DeliveryETA — Predictive Delivery-Time Modeling

DeliveryETA is a Machine Learning regression project that predicts the total delivery time of an order in minutes. The model uses information such as distance, traffic, weather, driver profile, vehicle type, order type, and order time.

## Business objective

Answer the following question for a new order:

> How many minutes will this delivery take?

The result can help the company provide realistic ETAs, plan drivers and routes, understand delay factors, and identify deliveries at risk of being late.

## Recommended project architecture

```text
delivery-eta/
├── app/
│   └── streamlit_app.py          # User interface and predictions
├── data/
│   ├── raw/
│   │   └── food_delivery.csv     # Original dataset (never modify it)
│   ├── processed/
│   │   └── delivery_clean.csv    # Clean dataset
│   └── README.md                 # Data source and column description
├── models/
│   └── delivery_eta_pipeline.joblib
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_optimization.ipynb
├── reports/
│   ├── figures/                  # EDA and model charts
│   └── model_results.csv         # Metrics for all models
├── src/
│   ├── __init__.py
│   ├── data_cleaning.py          # Cleaning functions
│   ├── feature_engineering.py    # Distance, order hour, etc.
│   ├── preprocessing.py          # ColumnTransformer and encoders
│   ├── train.py                  # Model training and comparison
│   ├── evaluate.py               # MAE, MSE, RMSE, R²
│   └── predict.py                # Load pipeline and predict
├── tests/
│   ├── test_data_cleaning.py
│   └── test_prediction.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Five-day implementation plan

### Day 1 — Monday, September 21, 2026: Setup and cleaning

- Create the repository and folder structure.
- Create and activate a Python virtual environment.
- Install the required packages.
- Download the dataset and place it in `data/raw/`.
- Inspect its shape, columns, data types, missing values, and duplicates.
- Remove unwanted text from values and convert columns to correct types.
- Handle missing values, duplicates, invalid GPS coordinates, and unrealistic values.
- Document and justify every cleaning decision.
- Export the clean dataset to `data/processed/delivery_clean.csv`.

**Deliverable:** clean dataset and `01_data_cleaning.ipynb`.

### Day 2 — Tuesday, September 22, 2026: EDA and feature engineering

- Calculate descriptive statistics for numerical columns.
- Calculate frequency tables for categorical columns.
- Plot the distribution of `Time_taken(min)`.
- Analyze delivery time against every important numerical and categorical feature.
- Build and interpret the correlation matrix.
- Create at least two useful features:
  - `delivery_distance_km` using restaurant and customer coordinates.
  - `order_hour` from the order time.
- Optionally create `driver_experience_group` or `is_peak_hour`.
- Save useful figures in `reports/figures/`.

**Deliverable:** completed `02_eda.ipynb` with interpreted charts and engineered features.

### Day 3 — Wednesday, September 23, 2026: Preprocessing and model comparison

- Separate features `X` from target `y`.
- Split the data into 80% training and 20% testing sets.
- Use `OneHotEncoder` for categorical variables.
- Use `StandardScaler` or `MinMaxScaler` for numerical variables.
- Combine preprocessing and each model in a scikit-learn `Pipeline`.
- Train at least four regression models:
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor
  - Gradient Boosting Regressor
- Compare MAE, MSE, RMSE, and R² using the same test set.
- Store the comparison in `reports/model_results.csv`.

**Deliverable:** `03_model_training.ipynb` and a justified choice of the best model.

### Day 4 — Thursday, September 24, 2026: Optimization and final evaluation

- Tune the best model with `RandomizedSearchCV` or `GridSearchCV`.
- Evaluate the optimized model with MAE, RMSE, R², and adjusted R².
- Compare training and test metrics to detect overfitting.
- Plot actual versus predicted delivery times and residual errors.
- Interpret results in business language, for example: “The model makes an average error of X minutes.”
- Save the complete fitted pipeline with `joblib` as `models/delivery_eta_pipeline.joblib`.
- Test prediction with one realistic example.

**Deliverable:** optimized and saved pipeline plus `04_model_optimization.ipynb`.

### Day 5 — Friday, September 25, 2026: Streamlit and final delivery

- Create the Streamlit form for distance or coordinates, traffic, weather, vehicle, driver rating, order time, and other required features.
- Load the saved pipeline and show the predicted ETA in minutes.
- Add pages or sections for:
  - dataset visualizations;
  - model metrics;
  - actual-versus-predicted performance;
  - an explanation of the prediction inputs.
- Validate inputs and display clear error messages.
- Test the complete application.
- Clean the repository and finalize the README.
- Make the final Git commit and prepare the demonstration.

**Deliverable:** working Streamlit application and documented Git repository.

## Environment setup

```bash
git clone <repository-url>
cd delivery-eta

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Suggested `requirements.txt` packages:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
streamlit
```

## Run the project

Train the model after implementing `src/train.py`:

```bash
python -m src.train
```

Launch the application:

```bash
streamlit run app/streamlit_app.py
```

## Evaluation metrics

| Metric | Meaning |
|---|---|
| MAE | Average absolute prediction error in minutes |
| MSE | Average squared error; penalizes large errors |
| RMSE | Typical error in minutes, with stronger penalty for large errors |
| R² | Proportion of variation in delivery time explained by the model |
| Adjusted R² | R² adjusted for the number of input features |

## Important rules

- Never modify the original file inside `data/raw/`.
- Fit preprocessing only on training data by using a `Pipeline`.
- Use the same train/test split when comparing models.
- Explain every cleaning, preprocessing, and modeling decision.
- Keep all code identifiers and feature names in English.
- Do not commit `.venv/`, cache files, or sensitive information.

## Final expected result

The user enters the characteristics of a new delivery, and the application returns a result such as:

```text
Estimated delivery time: 31 minutes
```