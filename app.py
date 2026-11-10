
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests

from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


# =====================================================
# 1. PAGE CONFIGURATION
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
# 2. CUSTOM CSS
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
}

.subtitle {
    color: #aab4c3;
    font-size: 16px;
}

.product-card {
    background: #1b2330;
    border: 1px solid #303b4b;
    border-radius: 14px;
    padding: 15px;
    margin-bottom: 15px;
}

.product-name {
    color: white;
    font-size: 16px;
    font-weight: bold;
    min-height: 65px;
}

.price {
    color: #42d392;
    font-size: 22px;
    font-weight: bold;
}

.similarity {
    color: #73b7ff;
    font-weight: bold;
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
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. LOAD DATASET AND ML MODEL
# =====================================================

@st.cache_resource
def load_project():

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "electronics_products.pkl not found!"
        )

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "product_cosine_similarity.pkl not found!"
        )

    df = pd.read_pickle(DATA_FILE)

    similarity = joblib.load(MODEL_FILE)

    df = df.reset_index(drop=True)

    if similarity.shape != (len(df), len(df)):
        raise ValueError(
            "Dataset and similarity matrix dimensions do not match."
        )

    required_columns = ["product_name"]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(
                f"Missing required column: {col}"
            )

    return df, similarity


# =====================================================
# 4. IMAGE URL CLEANING
# =====================================================

def clean_image_url(url):

    if not isinstance(url, str):
        return None

    url = url.strip()

    if not url.startswith(("http://", "https://")):
        return None

    parts = urlsplit(url)

    # Remove Amazon WEBP tracking path
    clean_path = (
        parts.path
        .replace("/W/WEBP_402378-T1", "")
        .replace("/W/WEBP_402378-T2", "")
    )

    return urlunsplit((
        parts.scheme,
        parts.netloc,
        clean_path,
        parts.query,
        ""
    ))


# =====================================================
# 5. FETCH PRODUCT IMAGES
# =====================================================

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_image(url):

    clean_url = clean_image_url(url)

    if not clean_url:
        return None

    try:

        response = requests.get(
            clean_url,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8"
            },
            timeout=12
        )

        if response.status_code != 200:
            return None

        content_type = response.headers.get(
            "Content-Type", ""
        )

        if not content_type.startswith("image/"):
            return None

        return response.content

    except requests.RequestException:
        return None


# =====================================================
# 6. DISPLAY IMAGE
# =====================================================

def display_image(url):

    image_data = fetch_image(url)

    if image_data:

        st.image(
            image_data,
            use_container_width=True
        )

    else:

        st.markdown("""
        <div style="
            height:190px;
            background:#273244;
            border-radius:12px;
            display:flex;
            align-items:center;
            justify-content:center;
            color:#ffffff;
            font-size:17px;
            text-align:center;
        ">
            📦 Product Image Unavailable
        </div>
        """, unsafe_allow_html=True)


# =====================================================
# 7. HELPER FUNCTIONS
# =====================================================

def clean_text(value):

    if pd.isna(value):
        return "Not available"

    value = str(value).strip()

    return value if value else "Not available"


def get_price(value):

    if pd.isna(value):
        return None

    try:

        value = str(value)

        value = (
            value.replace("₹", "")
            .replace(",", "")
            .replace(" ", "")
        )

        return float(value)

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


# =====================================================
# 8. RECOMMENDATION ENGINE
# =====================================================

def recommend_products(
    selected_index,
    df,
    similarity,
    top_n
):

    scores = similarity[selected_index]

    if hasattr(scores, "toarray"):
        scores = scores.toarray()

    scores = np.asarray(scores).flatten()

    if len(scores) != len(df):
        return pd.DataFrame()

    ranked_indices = np.argsort(scores)[::-1]

    recommendations = []

    for index in ranked_indices:

        if index == selected_index:
            continue

        score = float(scores[index])

        if score <= 0:
            continue

        product = df.iloc[index].to_dict()

        product["similarity"] = score * 100

        recommendations.append(product)

        if len(recommendations) >= top_n:
            break

    return pd.DataFrame(recommendations)


# =====================================================
# 9. HEADER
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
# 10. LOAD PROJECT
# =====================================================

try:

    with st.spinner("Loading ElectroFind..."):

        products, cosine_sim = load_project()

except Exception as error:

    st.error(f"Project Error: {error}")

    st.info(
        "Check karo ki dono PKL files app.py ke same folder mein hain."
    )

    st.stop()


# =====================================================
# 11. PREPARE DATA
# =====================================================

products["product_name"] = (
    products["product_name"]
    .fillna("Unnamed Product")
    .astype(str)
)

default_columns = {
    "category": "Other",
    "rating": np.nan,
    "discounted_price": np.nan,
    "actual_price": np.nan,
    "discount_percentage": np.nan,
    "img_link": None
}

for column, default in default_columns.items():

    if column not in products.columns:
        products[column] = default


# =====================================================
# 12. DASHBOARD METRICS
# =====================================================

total_products = len(products)

total_categories = products["category"].nunique()

average_rating = pd.to_numeric(
    products["rating"],
    errors="coerce"
).mean()

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Products",
    f"{total_products:,}"
)

col2.metric(
    "Categories",
    f"{total_categories:,}"
)

col3.metric(
    "Average Rating",
    f"{average_rating:.2f}/5"
    if pd.notna(average_rating)
    else "N/A"
)

st.divider()


# =====================================================
# 13. SIDEBAR
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
    placeholder="Search electronics..."
)

if search_query:

    filtered_products = filtered_products[
        filtered_products["product_name"].str.contains(
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
        key=lambda col: col.map(get_price),
        ascending=True,
        na_position="last"
    )

elif sort_option == "Price: High to Low":

    filtered_products = filtered_products.sort_values(
        by="discounted_price",
        key=lambda col: col.map(get_price),
        ascending=False,
        na_position="last"
    )

elif sort_option == "Rating: High to Low":

    filtered_products = filtered_products.sort_values(
        by="rating",
        key=lambda col: pd.to_numeric(
            col,
            errors="coerce"
        ),
        ascending=False,
        na_position="last"
    )

st.sidebar.caption(
    f"{len(filtered_products)} products found"
)


# =====================================================
# 14. PRODUCT SELECTION
# =====================================================

st.subheader("🛍️ Choose Your Product")

if filtered_products.empty:

    st.warning("No products found!")

    st.stop()

available_indices = filtered_products.index.tolist()

selected_index = st.selectbox(
    "Select Product",

    options=available_indices,

    format_func=lambda index: (
        products.iloc[index]["product_name"][:100]
    )
)

selected_product = products.iloc[selected_index]


# =====================================================
# 15. SELECTED PRODUCT DETAILS
# =====================================================

st.divider()

st.subheader("📌 Selected Product")

left, right = st.columns([1, 2])

with left:

    display_image(
        selected_product.get("img_link")
    )

with right:

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
        f"⭐ {rating}/5"
        if rating is not None
        else "Not available"
    )

    st.write(
        "**Actual Price:**",
        format_price(selected_product.get("actual_price"))
    )

    st.write(
        "**Discount:**",
        clean_text(
            selected_product.get("discount_percentage")
        )
    )

st.divider()


# =====================================================
# 16. RECOMMENDATION SETTINGS
# =====================================================

st.subheader("🤖 Smart Recommendations")

top_n = st.slider(
    "Number of recommendations",
    min_value=3,
    max_value=10,
    value=5
)


# =====================================================
# 17. SHOW RECOMMENDATIONS
# =====================================================

if st.button(
    "✨ Get Similar Products",
    type="primary",
    use_container_width=True
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
            "No similar products found."
        )

    else:

        st.success(
            f"{len(recommendations)} similar products found!"
        )

        st.subheader("🎯 Recommended Products")

        for start in range(
            0,
            len(recommendations),
            3
        ):

            columns = st.columns(3)

            batch = recommendations.iloc[
                start:start + 3
            ]

            for column, (_, product) in zip(
                columns,
                batch.iterrows()
            ):

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

                    st.caption(
                        clean_text(product.get("category"))
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
# 18. FOOTER
# =====================================================

st.divider()

st.markdown("""
<div class="footer">

    ⚡ ElectroFind

    <br>

    Smart Electronics Product Recommendation System

    <br>

    Built with Python | Pandas | Scikit-learn | Streamlit

    <br><br>

    © 2026 ElectroFind

</div>
""", unsafe_allow_html=True)
