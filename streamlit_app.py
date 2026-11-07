from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from backend.app.ml.dataset import (
    FAILURE_MODE_COLUMNS,
    FEATURE_COLUMNS,
    LABEL_COLUMNS,
    NUMERIC_SENSOR_COLUMNS,
    TARGET_COLUMN,
    normalize_column_names,
    validate_dataset,
)

st.set_page_config(page_title="SentinelML", page_icon="🏭", layout="wide")

st.title("SentinelML")
st.caption("Predictive-maintenance research and dataset exploration")
st.warning(
    "This demo uses the synthetic AI4I 2020 benchmark. Its labels mark failure "
    "events at the same recorded datapoint; this is not future-failure prediction "
    "or evidence of real-factory performance."
)

with st.sidebar:
    st.header("Dataset")
    st.markdown(
        "Upload the validated [AI4I 2020 CSV]"
        "(https://archive.ics.uci.edu/static/public/601/data.csv) "
        "to explore it. The file is processed in memory and is not saved."
    )
    uploaded_file = st.file_uploader("Upload AI4I CSV", type="csv")

if uploaded_file is None:
    st.info("Upload the AI4I 2020 CSV in the sidebar to get started.")
    st.stop()

try:
    frame = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as error:
    st.error(f"Could not read the uploaded CSV: {error}")
    st.stop()

frame = normalize_column_names(frame)
validation = validate_dataset(frame)
if not validation.is_valid:
    st.error("This file did not pass the AI4I dataset checks.")
    for error in validation.errors:
        st.write(f"- {error}")
    st.stop()

for warning in validation.warnings:
    st.warning(warning)

frame = frame.assign(
    **{
        column: pd.to_numeric(frame[column], errors="coerce")
        for column in LABEL_COLUMNS
    }
)

st.caption(f"Validated {validation.row_count:,} rows and {validation.column_count} columns.")

overview_tab, exploration_tab, dataset_tab = st.tabs(
    ["Overview", "Explore data", "Dataset"]
)

with overview_tab:
    failure_rate = validation.failure_rate
    non_failure_count = validation.row_count - validation.failure_count
    first, second, third = st.columns(3)
    first.metric("Records", f"{validation.row_count:,}")
    second.metric("Failure events", f"{validation.failure_count:,}")
    third.metric("Failure rate", f"{failure_rate:.2%}")

    st.subheader("Recorded labels")
    label_counts = pd.Series(
        {"No failure": non_failure_count, "Failure": validation.failure_count},
        name="Records",
    )
    st.bar_chart(label_counts)

    st.subheader("Failure-event rate by product type")
    failure_rate_by_type = (
        frame.groupby("Type", observed=True)[TARGET_COLUMN].mean().rename("Failure rate")
    )
    st.bar_chart(failure_rate_by_type)

with exploration_tab:
    st.subheader("Sensor measurements")
    first, second = st.columns(2)
    x_feature = first.selectbox("Horizontal axis", NUMERIC_SENSOR_COLUMNS)
    y_feature = second.selectbox(
        "Vertical axis", NUMERIC_SENSOR_COLUMNS, index=1
    )
    st.scatter_chart(
        frame,
        x=x_feature,
        y=y_feature,
        color=TARGET_COLUMN,
        height=420,
    )
    st.caption("Color indicates the recorded Machine failure label, not a model prediction.")

    selected_feature = st.selectbox("Feature distribution", FEATURE_COLUMNS)
    if selected_feature in NUMERIC_SENSOR_COLUMNS:
        distribution = frame[selected_feature].value_counts(bins=20).sort_index()
        st.bar_chart(distribution)
    else:
        st.bar_chart(frame[selected_feature].value_counts().sort_index())

with dataset_tab:
    st.subheader("Validation")
    st.json(validation.as_dict())

    st.subheader("Failure-mode labels")
    mode_counts = frame.loc[:, FAILURE_MODE_COLUMNS].sum().astype(int).rename("Events")
    st.bar_chart(mode_counts)
    st.caption(
        "Failure-mode columns describe labels and are excluded from model features."
    )

    st.subheader("Preview")
    st.dataframe(frame.head(100), width="stretch")
