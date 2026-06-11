import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

# -----------------------------
# PAGE SETTINGS
# -----------------------------
st.set_page_config(
    page_title="Delhi AQI AI Dashboard",
    page_icon="🌫️",
    layout="wide"
)


# -----------------------------
# DATA
# -----------------------------
@st.cache_data
def load_data():
    data = {
        "Month": [
            "January", "February", "March", "April",
            "May", "June", "July", "August",
            "September", "October", "November", "December"
        ],
        2018: [328, 243, 203, 222, 217, 202, 104, 111, 112, 269, 335, 360],
        2019: [328, 242, 184, 211, 221, 189, 134, 86, 98, 234, 312, 337],
        2020: [286, 241, 128, 110, 144, 123, 84, 64, 118, 265, 328, 332],
        2021: [324, 288, 223, 202, 144, 147, 110, 107, 78, 173, 377, 336],
        2022: [279, 225, 217, 255, 212, 190, 87, 93, 104, 210, 321, 319],
        2023: [311, 237, 170, 180, 171, 130, 84, 116, 108, 219, 373, 348],
        2024: [355, 218, 176, 182, 223, 179, 96, 72, 105, 234, 374, 294],
        2025: [306, 214, 170, 210, 170, 141, 78, 89, 105, 223, 354, 351]
    }

    return pd.DataFrame(data).set_index("Month")


df = load_data()

# -----------------------------
# TITLE
# -----------------------------
st.title("🌫️ Delhi AQI Analysis & AI Forecasting")
st.markdown(
    "### Data Science Passion Project\n"
    "Analysis of Delhi Air Quality Index trends from 2018–2025."
)

# -----------------------------
# METRICS
# -----------------------------
annual_avg = df.mean()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Average AQI", round(annual_avg.mean()))
col2.metric("Best Year", "2020")
col3.metric("Worst Year", "2024")
col4.metric("Worst Month", "November")

st.divider()

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Yearly Trends",
        "📅 Monthly Analysis",
        "🔥 Heatmap",
        "🤖 AI Prediction"
    ]
)

# -----------------------------
# TAB 1
# -----------------------------
with tab1:
    fig = px.bar(
        x=annual_avg.index,
        y=annual_avg.values,
        title="Annual Average AQI",
        labels={"x": "Year", "y": "AQI"},
        color=annual_avg.values,
        color_continuous_scale="Reds"
    )

    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# TAB 2
# -----------------------------
with tab2:
    year = st.selectbox(
        "Choose Year",
        list(df.columns),
        index=len(df.columns) - 1
    )

    fig2 = px.line(
        x=df.index,
        y=df[year],
        markers=True,
        title=f"Monthly AQI in {year}"
    )

    fig2.update_layout(
        xaxis_title="Month",
        yaxis_title="AQI"
    )

    st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# TAB 3
# -----------------------------
with tab3:
    fig3 = px.imshow(
        df,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="YlOrRd",
        title="AQI Heatmap"
    )

    st.plotly_chart(fig3, use_container_width=True)

# -----------------------------
# TAB 4
# -----------------------------
with tab4:
    rows = []

    for year in df.columns:
        for month_num, month in enumerate(df.index, start=1):
            rows.append({
                "year": year,
                "month": month_num,
                "aqi": df.loc[month, year]
            })

    ml_df = pd.DataFrame(rows)

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    model.fit(
        ml_df[["year", "month"]],
        ml_df["aqi"]
    )

    future = pd.DataFrame({
        "year": [2026] * 12,
        "month": list(range(1, 13))
    })

    predictions = model.predict(future)

    fig4 = go.Figure()

    fig4.add_trace(
        go.Scatter(
            x=df.index,
            y=df[2025],
            mode="lines+markers",
            name="2025 Actual"
        )
    )

    fig4.add_trace(
        go.Scatter(
            x=df.index,
            y=predictions,
            mode="lines+markers",
            name="2026 Predicted"
        )
    )

    fig4.update_layout(
        title="2025 vs Predicted 2026 AQI",
        xaxis_title="Month",
        yaxis_title="AQI"
    )

    st.plotly_chart(fig4, use_container_width=True)

    prediction_table = pd.DataFrame({
        "Month": df.index,
        "Predicted AQI 2026": predictions.round().astype(int)
    })

    st.dataframe(
        prediction_table,
        use_container_width=True
    )

st.divider()

st.caption(
    "Built using Python, Streamlit, Plotly and Scikit-Learn."
)
