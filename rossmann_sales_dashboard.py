"""Rossmann Sales Prediction Website.

Run in the VS Code terminal:
    pip install streamlit pandas numpy plotly scikit-learn xgboost
    streamlit run rossmann_sales_dashboard.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from xgboost import XGBRegressor


st.set_page_config(
    page_title="Predict Rossman Sales",
    page_icon="📈",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
TRAIN_FILE = BASE_DIR / "train.csv"
STORE_FILE = BASE_DIR / "store.csv"

INPUT_COLUMNS = [
    "Store",
    "DayOfWeek",
    "SchoolHoliday",
    "Promo",
    "StateHoliday",
    "StoreType",
    "Assortment",
    "Day",
    "Month",
    "Year",
    "CompetitionDistance",
    "CompetitionOpen",
    "WeekOfYear",
    "Promo2",
    "Promo2Open",
    "IsPromo2Month",
]

NUMERIC_COLUMNS = [
    "Store",
    "SchoolHoliday",
    "Promo",
    "Day",
    "Month",
    "Year",
    "CompetitionDistance",
    "CompetitionOpen",
    "WeekOfYear",
    "Promo2",
    "Promo2Open",
    "IsPromo2Month",
]

CATEGORICAL_COLUMNS = [
    "DayOfWeek",
    "StateHoliday",
    "StoreType",
    "Assortment",
]


def split_date(dataframe):
    dataframe = dataframe.copy()
    dataframe["Date"] = pd.to_datetime(dataframe["Date"])
    dataframe["Year"] = dataframe["Date"].dt.year
    dataframe["Month"] = dataframe["Date"].dt.month
    dataframe["Day"] = dataframe["Date"].dt.day
    dataframe["WeekOfYear"] = dataframe["Date"].dt.isocalendar().week.astype(int)
    return dataframe


def add_competition_features(dataframe):
    dataframe = dataframe.copy()

    dataframe["CompetitionOpen"] = (
        12
        * (dataframe["Year"] - dataframe["CompetitionOpenSinceYear"])
        + (
            dataframe["Month"]
            - dataframe["CompetitionOpenSinceMonth"]
        )
    )

    dataframe["CompetitionOpen"] = (
        dataframe["CompetitionOpen"]
        .clip(lower=0)
        .fillna(0)
    )

    return dataframe


def add_promotion_features(dataframe):
    dataframe = dataframe.copy()

    promo2_year = dataframe["Promo2SinceYear"].fillna(0)
    promo2_week = dataframe["Promo2SinceWeek"].fillna(0)

    dataframe["Promo2Open"] = (
        12 * (dataframe["Year"] - promo2_year)
        + (dataframe["WeekOfYear"] - promo2_week) / 4
    ).clip(lower=0) * dataframe["Promo2"]

    month_names = {
        1: "Jan",
        2: "Feb",
        3: "Mar",
        4: "Apr",
        5: "May",
        6: "Jun",
        7: "Jul",
        8: "Aug",
        9: "Sept",
        10: "Oct",
        11: "Nov",
        12: "Dec",
    }

    def is_promo_month(row):
        interval = row.get("PromoInterval", "")
        if pd.isna(interval):
            return 0

        months = str(interval).split(",")
        return int(
            row["Promo2"] == 1
            and month_names[int(row["Month"])] in months
        )

    dataframe["IsPromo2Month"] = dataframe.apply(
        is_promo_month,
        axis=1,
    )

    return dataframe


def prepare_features(dataframe, scaler, encoder, max_distance):
    dataframe = dataframe.copy()
    dataframe["CompetitionDistance"] = dataframe[
        "CompetitionDistance"
    ].fillna(max_distance * 2)

    dataframe[NUMERIC_COLUMNS] = scaler.transform(
        dataframe[NUMERIC_COLUMNS]
    )

    dataframe[CATEGORICAL_COLUMNS] = dataframe[
        CATEGORICAL_COLUMNS
    ].astype(str)

    encoded_values = encoder.transform(
        dataframe[CATEGORICAL_COLUMNS]
    )

    encoded_columns = encoder.get_feature_names_out(
        CATEGORICAL_COLUMNS
    )

    encoded_dataframe = pd.DataFrame(
        encoded_values,
        columns=encoded_columns,
        index=dataframe.index,
    )

    return pd.concat(
        [
            dataframe[NUMERIC_COLUMNS],
            encoded_dataframe,
        ],
        axis=1,
    )


@st.cache_resource
def train_latest_model():
    if not TRAIN_FILE.exists() or not STORE_FILE.exists():
        raise FileNotFoundError(
            "Both train.csv and store.csv must be in the same folder "
            "as this file."
        )

    rossmann_data = pd.read_csv(TRAIN_FILE)
    store_data = pd.read_csv(STORE_FILE)

    merged_data = rossmann_data.merge(
        store_data,
        how="left",
        on="Store",
    )

    merged_data = merged_data[merged_data["Open"] == 1].copy()
    merged_data = split_date(merged_data)
    merged_data = add_competition_features(merged_data)
    merged_data = add_promotion_features(merged_data)

    max_distance = merged_data["CompetitionDistance"].max()
    merged_data["CompetitionDistance"] = merged_data[
        "CompetitionDistance"
    ].fillna(max_distance * 2)

    scaler = MinMaxScaler()
    scaler.fit(merged_data[NUMERIC_COLUMNS])

    try:
        encoder = OneHotEncoder(
            sparse_output=False,
            handle_unknown="ignore",
        )
    except TypeError:
        encoder = OneHotEncoder(
            sparse=False,
            handle_unknown="ignore",
        )

    encoder.fit(
        merged_data[CATEGORICAL_COLUMNS].astype(str)
    )

    training_features = prepare_features(
        merged_data,
        scaler,
        encoder,
        max_distance,
    )

    # Latest XGBRegressor configuration from analysis.ipynb.
    model = XGBRegressor(
        n_jobs=-1,
        random_state=42,
        n_estimators=1000,
        learning_rate=0.2,
        max_depth=10,
        subsample=0.9,
        colsample_bytree=0.7,
    )

    model.fit(training_features, merged_data["Sales"])

    return (
        model,
        scaler,
        encoder,
        max_distance,
        store_data,
        merged_data,
    )


st.title("Predict Rossman Sales")
st.caption(
    "Sales prediction using the latest XGBRegressor model and "
    "hyperparameters from analysis.ipynb."
)

try:
    (
        model,
        scaler,
        encoder,
        max_distance,
        store_data,
        training_data,
    ) = train_latest_model()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()
except Exception as error:
    st.error(f"Model initialization failed: {error}")
    st.stop()


st.header("Enter Prediction Inputs")

with st.form("sales_prediction_form"):
    first_column, second_column = st.columns(2)

    with first_column:
        store = st.selectbox(
            "Store",
            options=sorted(store_data["Store"].unique()),
            index=0,
        )

        day_of_week = st.selectbox(
            "Day of Week",
            options=[1, 2, 3, 4, 5, 6, 7],
            index=3,
        )

        prediction_date = st.date_input(
            "Date",
            value=pd.Timestamp("2015-09-20").date(),
        )

        open_status = st.selectbox(
            "Open",
            options=[0, 1],
            index=1,
            format_func=lambda value: (
                "Closed" if value == 0 else "Open"
            ),
        )

    with second_column:
        promo = st.selectbox(
            "Promo",
            options=[0, 1],
            index=1,
            format_func=lambda value: (
                "No" if value == 0 else "Yes"
            ),
        )

        state_holiday = st.selectbox(
            "State Holiday",
            options=["0", "a", "b", "c"],
            index=0,
        )

        school_holiday = st.selectbox(
            "School Holiday",
            options=[0, 1],
            index=0,
            format_func=lambda value: (
                "No" if value == 0 else "Yes"
            ),
        )

    submitted = st.form_submit_button(
        "Predict Sales",
        type="primary",
    )

if submitted:
    sample_input = {
        "Store": store,
        "DayOfWeek": day_of_week,
        "Date": str(prediction_date),
        "Open": open_status,
        "Promo": promo,
        "StateHoliday": state_holiday,
        "SchoolHoliday": school_holiday,
    }

    input_dataframe = pd.DataFrame([sample_input])
    input_merged_dataframe = input_dataframe.merge(
        store_data,
        how="left",
        on="Store",
    )

    input_merged_dataframe = split_date(input_merged_dataframe)
    input_merged_dataframe = add_competition_features(
        input_merged_dataframe
    )
    input_merged_dataframe = add_promotion_features(
        input_merged_dataframe
    )

    if open_status == 0:
        prediction = 0.0
    else:
        prepared_input = prepare_features(
            input_merged_dataframe,
            scaler,
            encoder,
            max_distance,
        )
        prediction = float(model.predict(prepared_input)[0])
        prediction = max(0.0, prediction)

    st.success(f"Predicted Sales: {prediction:,.2f}")
    st.dataframe(input_dataframe, use_container_width=True)


st.header("Training Data Overview")

overview_columns = st.columns(4)
overview_columns[0].metric(
    "Training Rows",
    f"{len(training_data):,}",
)
overview_columns[1].metric(
    "Stores",
    f"{training_data['Store'].nunique():,}",
)
overview_columns[2].metric(
    "Average Sales",
    f"{training_data['Sales'].mean():,.0f}",
)
overview_columns[3].metric(
    "Maximum Sales",
    f"{training_data['Sales'].max():,.0f}",
)

with st.expander("Sales Distribution"):
    figure = px.histogram(
        training_data,
        x="Sales",
        nbins=50,
        title="Training Sales Distribution",
    )
    st.plotly_chart(figure, use_container_width=True)