
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from pathlib import Path
from functools import lru_cache

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="ElectroFind | Smart Product Finder",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "electronics_products.pkl"
MODEL_FILE = BASE_DIR / "product_cosine_similarity.pkl"

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0;
    }

    .subtitle {
        color: #aab4c3;
        font-size: 16px;
        margin-bottom: 25px;
    }

    .product-card {
        background: #1b2330;
        border: 1px solid #303b4b;
        border-radius: 14px;
        padding: 15px;
        margin-bottom: 12px;
    }

    .product-name {
        color: #ffffff;
        font-size: 17px;
        font-weight: 700;
        min-height: 55px;
    }

    .product-info {
        color: #c5ceda;
        font-size: 14px;
        line-height: 1.8;
    }

    .price {
        color: #42d392;
        font-size: 21px;
        font-weight: 800;
    }

    .similarity {
        color: #73b7ff;
        font-weight: 700;
    }

    div[data-testid="stMetric"] {
        background: #1b2330;
        border: 1px solid #303b4b;
        padding: 15px;
        border-radius: 12px;
    }

    .footer {
        text-align: center;
        color: #8792a2;
        padding: 25px;
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================
# LOAD DATA AND MODEL
# =====================================================

@st.cache_resource
def load_project():

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "electronics_products.pkl file repository mein nahi mili."
        )

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "product_cosine_similarity.pkl file repository mein nahi mili."
        )

    df = pd.read_pickle(DATA_FILE)
    similarity = joblib.load(MODEL_FILE)

    df = df.reset_index(drop=True)

    if similarity.shape[0] != len(df):
        raise ValueError(
            "Dataset aur similarity matrix ka size match nahi kar raha."
        )

    if similarity.shape[1] != len(df):
        raise ValueError(
            "Similarity matrix ka column size dataset se match nahi karta."
        )

    required_columns = ["product_name"]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Dataset mein '{column}' column missing hai.")

    return df, similarity


# =====================================================
# IMAGE FETCHING
# =====================================================

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_image(url):

    if not isinstance(url, str):
        return None

    url = url.strip()

    if not url.startswith("http"):
        return None

    try:
        response = requests.get(
            url,
            timeout=8,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")

        if not content_type.startswith("image/"):
            return None

        return response.content

    except requests.RequestException:
        return None


def display_image(url):

    image_data = fetch_image(url)

    if image_data:
        st.image(image_data, use_container_width=True)
    else:
        st.markdown(
            """
            <div style="
                background:#273244;
                height:180px;
                border-radius:10px;
                display:flex;
                align-items:center;
                justify-content:center;
                color:#c5ceda;
                font-size:16px;">
                📦 Image Not Available
            </div>
            """,
            unsafe_allow_html=True
        )


# =====================================================
# HELPER FUNCTIONS
# =====================================================

def clean_text(value, default="Not available"):

    if pd.isna(value):
        return default

    value = str(value).strip()

    return value if value else default


def get_price(value):

    if pd.isna(value):
        return None

    try:
        cleaned = (
            str(value)
            .replace("₹", "")
            .replace(",", "")
            .replace(" ", "")
        )

        return float(cleaned)

    except (ValueError, TypeError):
        return None


def format_price(value):

    price = get_price(value)

    if price is None:
        return "Price unavailable"

    return f"₹{price:,.0f}"


def get_rating(value):

    try:
        rating = float(value)

        if 0 <= rating <= 5:
            return rating

        return None

    except (ValueError, TypeError):
        return None


def get_similarity_scores(matrix, index):

    scores = matrix[index]

    if hasattr(scores, "toarray"):
        scores = scores.toarray()

    return np.asarray(scores).flatten()


# =====================================================
# RECOMMENDATION ENGINE
# =====================================================

def recommend_products(
    selected_index,
    df,
    similarity,
    top_n=5
):

    scores = get_similarity_scores(
        similarity,
        selected_index
    )

    if len(scores) != len(df):
        return pd.DataFrame()

    ranked_indices = np.argsort(scores)[::-1]

    recommendations = []

    for index in ranked_indices:

        # Selected product ko recommendation mein mat dikhao
        if index == selected_index:
            continue

        score = float(scores[index])

        # Zero similarity wale products skip karo
        if score <= 0:
            continue

        product = df.iloc[index].to_dict()

        product["similarity"] = score * 100
        product["original_index"] = index

        recommendations.append(product)

        if len(recommendations) >= top_n:
            break

    return pd.DataFrame(recommendations)


# =====================================================
# HEADER
# =====================================================

st.markdown(
    '<div class="main-title">⚡ ElectroFind</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Smart Electronics Product Recommendation System'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# =====================================================
# LOAD PROJECT
# =====================================================

try:

    with st.spinner("Loading ElectroFind..."):

        products, cosine_sim = load_project()

except Exception as error:

    st.error(f"Project loading error: {error}")

    st.info(
        "Check karo ki dono PKL files aur app.py "
        "GitHub repository ke same root folder mein hain."
    )

    st.stop()


# =====================================================
# PREPARE DATA
# =====================================================

products["product_name"] = (
    products["product_name"]
    .fillna("Unnamed Product")
    .astype(str)
)

if "category" not in products.columns:
    products["category"] = "Other"

if "rating" not in products.columns:
    products["rating"] = np.nan

if "discounted_price" not in products.columns:
    products["discounted_price"] = np.nan

if "img_link" not in products.columns:
    products["img_link"] = None

if "actual_price" not in products.columns:
    products["actual_price"] = np.nan

if "discount_percentage" not in products.columns:
    products["discount_percentage"] = np.nan


# =====================================================
# DASHBOARD METRICS
# =====================================================

total_products = len(products)

total_categories = products["category"].nunique()

average_rating = pd.to_numeric(
    products["rating"],
    errors="coerce"
).mean()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Products",
        f"{total_products:,}"
    )

with col2:
    st.metric(
        "Categories",
        f"{total_categories:,}"
    )

with col3:
    st.metric(
        "Average Rating",
        f"{average_rating:.2f} / 5"
        if pd.notna(average_rating)
        else "N/A"
    )

st.divider()


# =====================================================
# SIDEBAR FILTERS
# =====================================================

st.sidebar.title("🔎 Find Products")

categories = sorted(
    products["category"]
    .fillna("Other")
    .astype(str)
    .unique()
    .tolist()
)

selected_category = st.sidebar.selectbox(
    "Select Category",
    ["All Categories"] + categories
)

filtered_products = products.copy()

if selected_category != "All Categories":

    filtered_products = filtered_products[
        filtered_products["category"].astype(str)
        == selected_category
    ]

search_query = st.sidebar.text_input(
    "Search Product",
    placeholder="Enter product name..."
)

if search_query:

    filtered_products = filtered_products[
        filtered_products["product_name"]
        .str.contains(
            search_query,
            case=False,
            na=False
        )
    ]

sort_option = st.sidebar.selectbox(
    "Sort Products",
    [
        "Default",
        "Price: Low to High",
        "Price: High to Low",
        "Rating: High to Low"
    ]
)

if sort_option == "Price: Low to High":

    filtered_products = filtered_products.sort_values(
        by="discounted_price",
        key=lambda column: column.map(get_price),
        ascending=True,
        na_position="last"
    )

elif sort_option == "Price: High to Low":

    filtered_products = filtered_products.sort_values(
        by="discounted_price",
        key=lambda column: column.map(get_price),
        ascending=False,
        na_position="last"
    )

elif sort_option == "Rating: High to Low":

    filtered_products = filtered_products.sort_values(
        by="rating",
        key=lambda column: pd.to_numeric(
            column,
            errors="coerce"
        ),
        ascending=False,
        na_position="last"
    )

st.sidebar.caption(
    f"{len(filtered_products)} products found"
)


# =====================================================
# PRODUCT SELECTION
# =====================================================

st.subheader("🛍️ Choose Your Product")

if filtered_products.empty:

    st.warning(
        "No products found. Try another search or category."
    )

    st.stop()


# Use original dataframe indices for similarity matrix
available_indices = filtered_products.index.tolist()

selected_index = st.selectbox(
    "Select a product to get similar recommendations",

    options=available_indices,

    format_func=lambda index: (
        products.iloc[index]["product_name"][:100]
    )
)


# =====================================================
# SELECTED PRODUCT DETAILS
# =====================================================

selected_product = products.iloc[selected_index]

st.divider()

st.subheader("📌 Selected Product")

product_col1, product_col2 = st.columns([1, 2])

with product_col1:

    display_image(
        selected_product.get("img_link")
    )

with product_col2:

    st.markdown(
        f"### {clean_text(selected_product['product_name'])}"
    )

    st.write(
        "**Category:**",
        clean_text(selected_product.get("category"))
    )

    st.markdown(
        f'<div class="price">'
        f'{format_price(selected_product.get("discounted_price"))}'
        f'</div>',
        unsafe_allow_html=True
    )

    rating = get_rating(
        selected_product.get("rating")
    )

    st.write(
        "**Rating:**",
        f"⭐ {rating}/5" if rating is not None else "Not available"
    )

    st.write(
        "**Original Price:**",
        format_price(selected_product.get("actual_price"))
    )

    discount = clean_text(
        selected_product.get("discount_percentage")
    )

    st.write(
        "**Discount:**",
        discount
    )

st.divider()


# =====================================================
# RECOMMENDATION SETTINGS
# =====================================================

st.subheader("🤖 Smart Recommendations")

top_n = st.slider(
    "Number of recommendations",
    min_value=3,
    max_value=10,
    value=5
)

if st.button(
    "✨ Get Similar Products",
    use_container_width=True,
    type="primary"
):

    with st.spinner("Finding similar products..."):

        recommendations = recommend_products(
            selected_index,
            products,
            cosine_sim,
            top_n
        )

    if recommendations.empty:

        st.warning(
            "No similar products found for this selection."
        )

    else:

        st.success(
            f"{len(recommendations)} similar products found!"
        )

        st.subheader("🎯 Recommended Products")

        # Display products in 3-column layout
        for start in range(0, len(recommendations), 3):

            columns = st.columns(3)

            batch = recommendations.iloc[start:start + 3]

            for column, (_, product) in zip(columns, batch.iterrows()):

                with column:

                    st.markdown(
                        '<div class="product-card">',
                        unsafe_allow_html=True
                    )

                    display_image(
                        product.get("img_link")
                    )

                    st.markdown(
                        f'<div class="product-name">'
                        f'{clean_text(product.get("product_name"))}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        f'<div class="product-info">'
                        f'Category: {clean_text(product.get("category"))}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        f'<div class="price">'
                        f'{format_price(product.get("discounted_price"))}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                    rating = get_rating(
                        product.get("rating")
                    )

                    st.write(
                        f"⭐ {rating}/5"
                        if rating is not None
                        else "⭐ Rating unavailable"
                    )

                    score = float(product["similarity"])

                    st.markdown(
                        f'<div class="similarity">'
                        f'Similarity: {score:.2f}%'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                    st.progress(
                        min(max(score / 100, 0.0), 1.0)
                    )

                    st.markdown(
                        '</div>',
                        unsafe_allow_html=True
                    )


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        ⚡ ElectroFind — Smart Electronics Recommendation System
        <br>
        Built using Python, Pandas, Scikit-learn and Streamlit
        <br><br>
        © 2026 ElectroFind
    </div>
    """,
    unsafe_allow_html=True
)
