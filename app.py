import pandas as pd
import streamlit as st
from xgboost import XGBRegressor

st.set_page_config(page_title="Demand Forecast", page_icon="📈")

# Must match the exact order used when training the model (features2)
FEATURES = [
    "store", "item", "dow", "month", "day", "year",
    "lag_14", "lag_28", "lag_364", "roll_mean_28",
    "lag_21", "lag_35", "lag_42", "dow_mean_4w", "roll_std_28",
]


def make_features(d: pd.DataFrame) -> pd.DataFrame:
    d = d.sort_values("date").copy()
    d["dow"] = d["date"].dt.dayofweek
    d["month"] = d["date"].dt.month
    d["day"] = d["date"].dt.day
    d["year"] = d["date"].dt.year

    s = d["sales"]
    for k in [14, 21, 28, 35, 42, 364]:
        d[f"lag_{k}"] = s.shift(k)
    d["roll_mean_28"] = s.shift(14).rolling(28).mean()
    d["roll_std_28"] = s.shift(14).rolling(28).std()
    d["dow_mean_4w"] = d[["lag_14", "lag_21", "lag_28", "lag_35"]].mean(axis=1)
    return d


@st.cache_resource
def load():
    model = XGBRegressor()
    model.load_model("model.json")
    hist = pd.read_csv("history.csv", parse_dates=["date"])
    return model, hist


model, hist = load()

st.title("📈 Demand Forecast: next 14 days")
st.caption(
    "XGBoost model trained on 5 years of daily sales (10 stores x 50 items). "
    "Demo only: history ends 2017-12-31, so the forecast covers the first "
    "14 days of January 2018."
)

col1, col2 = st.columns(2)
store = col1.selectbox("Store", sorted(hist["store"].unique()))
item = col2.selectbox("Item", sorted(hist["item"].unique()))

if st.button("Forecast"):
    s = hist[(hist["store"] == store) & (hist["item"] == item)].sort_values("date")
    last = s["date"].max()

    fut = pd.DataFrame(
        {"date": pd.date_range(last + pd.Timedelta(days=1), periods=14)}
    )
    fut["store"], fut["item"], fut["sales"] = store, item, float("nan")

    d = make_features(pd.concat([s, fut], ignore_index=True))
    f = d[d["date"] > last].copy()
    f["forecast"] = model.predict(f[FEATURES]).round(0)

    chart = pd.concat(
        [
            s.tail(60).set_index("date")["sales"].rename("actual"),
            f.set_index("date")["forecast"],
        ],
        axis=1,
    )
    st.line_chart(chart)

    st.metric("Total forecast (14 days)", int(f["forecast"].sum()))
    out = f[["date", "forecast"]].reset_index(drop=True)
    out["date"] = out["date"].dt.strftime("%Y-%m-%d (%a)")
    st.dataframe(out, use_container_width=True)
