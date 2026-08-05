"""Streamlit demonstration app (Section 5.4).

Upload a lateral-well CSV and its paired typewell CSV; the app locates the
Prediction-Start boundary, builds the same five features used at training time,
and predicts TVT for every masked row.

Usage:
    streamlit run app/app.py
"""

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import FEATURES, MODEL_PATH  # noqa: E402
from features import extract_features_for_well, get_prediction_start_index  # noqa: E402

st.set_page_config(page_title="ROGII Wellbore TVT Predictor", layout="wide")

st.title("ROGII Wellbore Geology Prediction")
st.markdown(
    "This app predicts the True Vertical Thickness (TVT) for the toe-end of a "
    "horizontal well based on its trajectory and the reference Gamma Ray (GR) "
    "signature from a paired vertical typewell."
)

st.sidebar.header("Upload Well Data")
horz_file = st.sidebar.file_uploader("Upload Horizontal Well CSV", type=["csv"])
type_file = st.sidebar.file_uploader("Upload Typewell CSV", type=["csv"])


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


model = load_model()
if model is None:
    st.warning(
        "⚠️ Trained model not found. Run `python src/train_model.py` first to "
        "generate `models/lgbm_model.joblib`."
    )
else:
    st.sidebar.success("✅ Model loaded successfully!")

if not (horz_file and type_file and model):
    st.info("Please upload both CSV files from the sidebar to generate a prediction.")
    st.stop()

horz_df = pd.read_csv(horz_file)
type_df = pd.read_csv(type_file)

st.subheader("Data Overview")
col1, col2 = st.columns(2)
with col1:
    st.write("**Horizontal Well (Lateral)**")
    st.dataframe(horz_df.head())
with col2:
    st.write("**Typewell (Vertical Reference)**")
    st.dataframe(type_df.head())

st.subheader("Prediction Results")

ps_idx = get_prediction_start_index(horz_df)
st.write(f"**Prediction Start (PS) index found at row {ps_idx}.**")

predict_df = extract_features_for_well("uploaded", horz_df, type_df, is_train=False)

if len(predict_df) == 0:
    st.success("The uploaded horizontal well has no missing TVT values to predict!")
    st.stop()

# The model predicts dTVT, the offset from the PS anchor; add ps_TVT back to
# recover absolute TVT.
predict_df["Predicted_TVT"] = predict_df["ps_TVT"] + model.predict(predict_df[FEATURES])

st.write(f"Generated predictions for **{len(predict_df)}** masked rows.")
st.dataframe(predict_df[["MD", "Z", "GR", "Predicted_TVT"]].head(10))

st.download_button(
    label="Download Predictions as CSV",
    data=predict_df[["MD", "Predicted_TVT"]].to_csv(index=False),
    file_name="predicted_tvt.csv",
    mime="text/csv",
)

st.subheader("Wellbore Trajectory Visualization")
known_df = horz_df.loc[:ps_idx]

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(
    known_df["MD"],
    known_df["TVT_input"],
    label="Known TVT (Before PS)",
    color="blue",
    linewidth=2,
)
ax.plot(
    predict_df["MD"],
    predict_df["Predicted_TVT"],
    label="Predicted TVT (After PS)",
    color="red",
    linestyle="--",
    linewidth=2,
)
ax.set_xlabel("Measured Depth (MD)")
ax.set_ylabel("True Vertical Thickness (TVT)")
ax.set_title("Predicted TVT along the Horizontal Well")
ax.invert_yaxis()  # TVT increases downwards
ax.legend()
ax.grid(True)

st.pyplot(fig)
