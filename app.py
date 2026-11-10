
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="ElectroFind",
    page_icon="⚡",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
.stApp {
    background-color: #101827;
    color: white;
}
h1 {
    color: #38bdf8;
    text-align: center;
}
.subtitle {
    text-align: center;
    color: #cbd5e1;
    font-size: 18px;
}
.product-card {
    background-color: #1e293b;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)

# Load ML model and dataset
BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_data():
    products = pd.read_pickle(
        BASE_DIR / "electronics_products.pkl"
    )

    similarity = joblib.load(
        BASE_DIR / "product_cosine_similarity.pkl"
    )

    products = products.reset_index(drop=True)

    if len(products) != similarity.shape[0]:
        raise ValueError("Dataset and similarity matrix do not match.")

    return products, similarity


try:
    products, cosine_sim = load_data()
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()


# Recommendation function
def recommend_products(product_name, top_n=5):

    matches = products[
        products["product_name"].str.lower()
        == product_name.lower()
    ]

    if matches.empty:
        return pd.DataFrame()

    product_index = matches.index[0]

    scores = list(enumerate(cosine_sim[product_index]))

    scores = sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )

    scores = [
        item for item in scores
        if item[0] != product_index
    ][:top_n]

    recommendations = []

    for index, score in scores:

        product = products.iloc[index]

        recommendations.append({
            "Product Name": product["product_name"],
            "Category": product["category"],
            "Price": product["discounted_price"],
            "Rating": product["rating"],
            "Similarity": round(float(score) * 100, 2)
        })

    return pd.DataFrame(recommendations)


# Main dashboard
st.title("⚡ ElectroFind")

st.markdown(
    '<p class="subtitle">Smart Electronics Product Recommendation System</p>',
    unsafe_allow_html=True
)

st.divider()

# Dashboard metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Products", len(products))

with col2:
    st.metric(
        "Categories",
        products["category"].nunique()
    )

with col3:
    st.metric("Recommendation Model", "TF-IDF")

st.divider()

# Product selection
st.subheader("🔍 Find Similar Products")

product_list = sorted(
    products["product_name"].dropna().unique().tolist()
)

selected_product = st.selectbox(
    "Select an electronics product",
    product_list
)

top_n = st.slider(
    "Number of recommendations",
    min_value=1,
    max_value=10,
    value=5
)

if st.button("🚀 Get Recommendations", use_container_width=True):

    result = recommend_products(
        selected_product,
        top_n
    )

    if result.empty:
        st.warning("No recommendations found.")

    else:
        st.success(
            f"Recommendations for: {selected_product}"
        )

        for index, row in result.iterrows():

            with st.container(border=True):

                st.subheader(row["Product Name"])

                c1, c2 = st.columns(2)

                with c1:
                    st.write("**Category:**", row["Category"])
                    st.write("**Price:**", row["Price"])

                with c2:
                    st.write("**Rating:**", row["Rating"])
                    st.write(
                        "**Similarity:**",
                        f'{row["Similarity"]}%'
                    )

                st.progress(
                    min(row["Similarity"] / 100, 1.0)
                )

st.divider()

st.caption(
    "ElectroFind | Machine Learning Based Product Recommendation System"
)
