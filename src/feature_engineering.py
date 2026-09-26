import numpy as np
import pandas as pd


def calculate_distance(dataframe):
    """
    Calculate the distance between the restaurant and delivery
    location using the Haversine formula.
    """

    earth_radius_km = 6371

    restaurant_latitude = np.radians(
        dataframe['Restaurant_latitude']
    )

    restaurant_longitude = np.radians(
        dataframe["Restaurant_longitude"]
    )

    delivery_latitude = np.radians(
        dataframe["Delivery_location_latitude"]
    )

    delivery_longitude = np.radians(
        dataframe["Delivery_location_longitude"]
    )

    latitude_difference = (
        delivery_latitude - restaurant_latitude
    )

    longitude_difference = (
        delivery_longitude - restaurant_longitude
    )

    haversine_value = (
        np.sin(latitude_difference / 2) ** 2
        + np.cos(restaurant_latitude)
        * np.cos(delivery_latitude)
        * np.sin(longitude_difference / 2) ** 2
    )

    haversine_value = np.clip(
        haversine_value,
        0,
        1
    )

    distance = (
        2
        * earth_radius_km
        * np.arcsin(np.sqrt(haversine_value))
    )

    return distance


def convert_time_to_minutes(time_column):
    """
    Convert a time column into the number of minutes since midnight.
    Example: 10:30:00 becomes 630 minutes.
    """

    parsed_time = pd.to_datetime(
        time_column,
        format="%H:%M:%S",
        errors="coerce"
    )

    return (
        parsed_time.dt.hour * 60
        + parsed_time.dt.minute
        + parsed_time.dt.second / 60
    )

def add_features(dataframe):
    """
    Create new features for delivery-time prediction.
    """

    dataframe = dataframe.copy()

    # Feature 1: gps distance is kilometres
    dataframe["distance_km"] = calculate_distance(
        dataframe
    )

    # convert order and pickup times into minutes
    order_minutes = convert_time_to_minutes(
        dataframe["Time_Orderd"]
    )

    pickup_minutes = convert_time_to_minutes(
        dataframe["Time_Order_picked"]
    )

    # Feature 2: hour when the order was placed
    dataframe["order_hour"] = (
        order_minutes // 60
    ).astype("int64")


    # Feature 3: preparation time 
    # modulo 1440 handels orders passing midnight
    dataframe["preparation_time_minutes"]  = (
        pickup_minutes - order_minutes
    ) % 1440

    return dataframe