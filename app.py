
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="ElectroFind | Smart Recommendations",
    page_icon="⚡",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

# ==========================================
# 2. CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.stApp {
    background-color: #0f172a;
    color: white;
}

header {
    background-color: transparent !important;
}

h1 {
    color: #38bdf8;
    text-align: center;
    font-size: 45px !important;
}

h2, h3 {
    color: #e2e8f0;
}

.subtitle {
    text-align: center;
    color: #94a3b8;
    font-size: 18px;
    margin-bottom: 30px;
}

div[data-testid="stMetric"] {
    background-color: #1e293b;
    padding: 18px;
    border-radius: 12px;
}

div.stButton > button {
    background-color: #0284c7;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 12px;
    font-weight: bold;
}

div.stButton > button:hover {
    background-color: #0369a1;
    color: white;
}

.product-card {
    background-color: #1e293b;
    padding: 15px;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. LOAD DATASET AND ML MODEL
# ==========================================

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
        raise ValueError(
            "Dataset and similarity matrix do not match."
        )

    return products, similarity


try:

    products, cosine_sim = load_data()

except Exception as error:

    st.error(f"Model loading error: {error}")
    st.stop()


# ==========================================
# 4. HELPER FUNCTIONS
# ==========================================

def get_image_url(product):

    image = product.get("img_link", None)

    if isinstance(image, str) and image.startswith("http"):
        return image

    return None


def recommend_products(product_name, top_n=5):

    matches = products[
        products["product_name"].astype(str).str.lower()
        == product_name.lower()
    ]

    if matches.empty:
        return pd.DataFrame()

    product_index = matches.index[0]

    scores = list(
        enumerate(cosine_sim[product_index])
    )

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

            "Product Name": product.get("product_name"),
            "Category": product.get("category"),
            "Price": product.get("discounted_price"),
            "Rating": product.get("rating"),
            "Similarity": round(float(score) * 100, 2),
            "Image": get_image_url(product)

        })

    return pd.DataFrame(recommendations)


# ==========================================
# 5. HEADER
# ==========================================

st.markdown("<h1>⚡ ElectroFind</h1>", unsafe_allow_html=True)

st.markdown(
    '<p class="subtitle">Smart Electronics Product Recommendation System</p>',
    unsafe_allow_html=True
)

st.divider()


# ==========================================
# 6. DASHBOARD METRICS
# ==========================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Products",
        len(products)
    )

with col2:
    st.metric(
        "Product Categories",
        products["category"].nunique()
    )

with col3:
    st.metric(
        "Recommendation Algorithm",
        "TF-IDF"
    )

st.divider()


# ==========================================
# 7. SEARCH AND CATEGORY FILTER
# ==========================================

st.subheader("🔍 Explore Electronics Products")

categories = sorted(
    products["category"].dropna().unique().tolist()
)

selected_category = st.selectbox(
    "Select Product Category",
    ["All Categories"] + categories
)

if selected_category == "All Categories":

    filtered_products = products

else:

    filtered_products = products[
        products["category"] == selected_category
    ]

product_names = sorted(
    filtered_products["product_name"]
    .dropna()
    .unique()
    .tolist()
)

if not product_names:

    st.warning("No products found in this category.")
    st.stop()

selected_product = st.selectbox(
    "Choose a Product",
    product_names
)


# ==========================================
# 8. SELECTED PRODUCT DETAILS
# ==========================================

selected_row = products[
    products["product_name"] == selected_product
].iloc[0]

st.markdown("### 🛍️ Selected Product")

col1, col2 = st.columns([1, 2])

with col1:

    image_url = get_image_url(selected_row)

    if image_url:
        st.image(image_url, use_container_width=True)
    else:
        st.info("Product image not available.")

with col2:

    st.subheader(selected_row["product_name"])

    st.write("**Category:**", selected_row.get("category", "N/A"))

    st.write(
        "**Price:**",
        selected_row.get("discounted_price", "N/A")
    )

    st.write(
        "**Rating:**",
        selected_row.get("rating", "N/A")
    )

    st.write(
        "**Description:**",
        selected_row.get("about_product", "No description available")
    )


st.divider()


# ==========================================
# 9. RECOMMENDATION SETTINGS
# ==========================================

st.subheader("🤖 AI Product Recommendations")

top_n = st.slider(
    "Number of recommendations",
    min_value=1,
    max_value=10,
    value=5
)

if st.button(
    "🚀 Get Similar Products",
    use_container_width=True
):

    result = recommend_products(
        selected_product,
        top_n
    )

    if result.empty:

        st.warning("No recommendations found.")

    else:

        st.success(
            f"Found {len(result)} similar products!"
        )

        st.markdown("### Recommended Products")

        # ==================================
        # 10. DISPLAY PRODUCT CARDS
        # ==================================

        for start in range(0, len(result), 3):

            cols = st.columns(3)

            for col, (_, row) in zip(
                cols,
                result.iloc[start:start + 3].iterrows()
            ):

                with col:

                    with st.container(border=True):

                        if row["Image"]:

                            st.image(
                                row["Image"],
                                use_container_width=True
                            )

                        else:

                            st.info("Image unavailable")

                        st.markdown(
                            f"**{row['Product Name']}**"
                        )

                        st.write(
                            "Category:",
                            row["Category"]
                        )

                        st.write(
                            "Price:",
                            row["Price"]
                        )

                        st.write(
                            "Rating:",
                            row["Rating"]
                        )

                        st.write(
                            "Similarity:",
                            f"{row['Similarity']}%"
                        )

                        st.progress(
                            min(row["Similarity"] / 100, 1.0)
                        )


# ==========================================
# 11. FOOTER
# ==========================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#94a3b8;">
        <h3>⚡ ElectroFind</h3>
        <p>Machine Learning Based Product Recommendation System</p>
        <p>Built using Python, Streamlit and TF-IDF</p>
    </div>
    """,
    unsafe_allow_html=True
)
