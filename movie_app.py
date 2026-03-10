import streamlit as st
import joblib
import pandas as pd
import math

# --------------------------------
# Page Config
# --------------------------------

st.set_page_config(
    page_title="Movie Success Predictor",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Movie Success Predictor")
st.markdown(
"""
Predicting film performance using machine learning.

Inputs:
- Bechdel test rating
- Genre combinations
- MPAA rating
"""
)

# --------------------------------
# Constants
# --------------------------------

GENRES = [
"History","Romance","Family","Mystery","Horror","Biography","Crime",
"Thriller","Western","Sport","Music","SciFi","Adventure","Drama",
"Comedy","Musical","Fantasy","Action","War","Animation","Documentary",
"Adult"
]

MPAA_RATINGS = ['g','pg_13','r','nc_17','approved','not_rated']

# --------------------------------
# Feature Construction
# --------------------------------

genre_features = [f"genre_{g}" for g in GENRES]
interaction_features = [f"{g}_x_bechdel" for g in genre_features]

FEATURES = (
    genre_features +
    ["bechdel"] +
    MPAA_RATINGS +
    interaction_features
)

# --------------------------------
# Load Models
# --------------------------------

@st.cache_resource
def load_models():
    return {
        "DEA Efficiency": joblib.load("movie_model.pkl"),
        "Domestic Profit ROI": joblib.load("domestic_model.pkl"),
        "International Profit ROI": joblib.load("international_model.pkl"),
        "IMDb Rating": joblib.load("imdb_model.pkl"),
        "Awards Won": joblib.load("awards_model.pkl")
    }

models = load_models()

# --------------------------------
# User Inputs
# --------------------------------

bechdel = st.slider("Bechdel Rating", 0, 3, 1)
genres = st.multiselect(
    "Genres",
    GENRES
)
rating = st.selectbox(
    "MPAA Rating",
    ["g", "pg","pg_13","r","nc_17","approved","not_rated"]
)
prediction_type = st.selectbox(
    "Prediction Target",
    list(models.keys())
)

model = models[prediction_type]

# --------------------------------
# Build Input Data
# --------------------------------

def build_input(bechdel, genres, rating):
    data = {f:0 for f in FEATURES}
    data["bechdel"] = bechdel
    # genre encoding
    for g in GENRES:
        data[f"genre_{g}"] = int(g in genres)

    # interaction terms
    for g in GENRES:
        data[f"genre_{g}_x_bechdel"] = data[f"genre_{g}"] * bechdel

    # mpaa
    data[rating] = 1
    return pd.DataFrame([data])[FEATURES]

# --------------------------------
# Prediction
# --------------------------------

if st.button("Predict"):
    input_df = build_input(bechdel, genres, rating)
    prediction = model.predict(input_df)[0]
    if prediction_type == "Awards Won":
        prediction = math.exp(1 + prediction)
    st.subheader("Prediction Result")
    st.metric(
        label=prediction_type,
        value=f"{prediction:.2f}"
    )

