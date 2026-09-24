import pandas as pd
import numpy as np

def clean_data(dataframe):
    dataframe = dataframe.copy()

    # 1 clean column names 
    dataframe.columns = dataframe.columns.str.strip()

    # 2. strip whitespace on all text columns 
    text_columns = [
        "Weatherconditions",
        "Road_traffic_density",
        "Type_of_vehicle",
        "City",
    ]

    for column in text_columns:
        dataframe[column] = dataframe[column].astype(str).str.strip()


    #    remove "parasite text " glued to real values
    #    "(min) 24"          -> 24
    #    "conditions Sunny"  -> "Sunny"

    dataframe["Time_taken(min)"] = (
         dataframe["Time_taken(min)"]
        .astype(str)
        .str.extract(r"(\d+)")[0]
    )

    dataframe["Weatherconditions"] = (
        dataframe["Weatherconditions"]
        .str.replace("conditions", "", regex=False)
        .str.strip()
    )

    # 4. turn the literal text "NaN" into real missing values
    dataframe = dataframe.replace(
        {"NaN": np.nan, "nan": np.nan, "": np.nan}
    )


    # 5. convert columns to the correct numeric types
    numeric_columns = [
        "Restaurant_latitude",
        "Restaurant_longitude",
        "Delivery_location_latitude",
        "Delivery_location_longitude",
        "Time_taken(min)",
    ]
    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce"
        )

    # 6. convert date/time columns
    dataframe["Time_Orderd"] = pd.to_datetime(
        dataframe["Time_Orderd"], format="%H:%M:%S", errors="coerce"
    ).dt.time
    dataframe["Time_Order_picked"] = pd.to_datetime(
        dataframe["Time_Order_picked"], format="%H:%M:%S", errors="coerce"
    ).dt.time

    # 7. remove duplicate rows (checked on ID, which should be unique)
    dataframe = dataframe.drop_duplicates(subset=["ID"])

    # 8. remove rows with a missing target
    dataframe = dataframe.dropna(subset=["Time_taken(min)"])

    # 9. fix invalid / impossible value -\, convert to missing

    coord_columns = [
        "Restaurant_latitude",
        "Restaurant_longitude",
        "Delivery_location_latitude",
        "Delivery_location_longitude",
    ] 

    for column in coord_columns:
        dataframe[column] = dataframe[column].abs()


    # Drop these rows: a GPS location can't be reasonably imputed.

    zero_location = (
        (dataframe["Restaurant_latitude"] < 0.01)
        & (dataframe["Restaurant_longitude"] < 0.01)
    )

    dataframe= dataframe[~zero_location]




    # Convert invalid delivery times to missing
    dataframe.loc[
        dataframe["Time_taken(min)"] <= 0,
        "Time_taken(min)"
    ] = np.nan

    # Remove rows with missing target or times
    dataframe = dataframe.dropna(
        subset=[
            "Time_taken(min)",
            "Time_Orderd",
            "Time_Order_picked",
        ]
    )

    # Fill categorical missing values with mode
    categorical_missing_columns = [
        "Weatherconditions",
        "Road_traffic_density",
        "City",
    ]

    for column in categorical_missing_columns:
        dataframe[column] = dataframe[column].fillna(
            dataframe[column].mode().iloc[0]
        )

    # Remove unhelpful columns
    columns_to_drop = [
        "ID",
        "Delivery_person_Age",
        "Delivery_person_Ratings",
        "Order_Date",
        "multiple_deliveries",
        "Type_of_order",
        "Delivery_person_ID",
        "Vehicle_condition",
        "Festival",
    ]

    dataframe = dataframe.drop(
        columns=columns_to_drop,
        errors="ignore"
    )

    dataframe = dataframe.reset_index(drop=True)

    return dataframe