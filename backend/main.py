
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from functools import lru_cache
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="Electronics Product Recommendation System",
    version="1.0.0"
)

BASE_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def load_model():
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


class RecommendationRequest(BaseModel):
    product_name: str
    top_n: int = 5


def safe_value(value):
    if pd.isna(value):
        return None

    if isinstance(value, np.generic):
        return value.item()

    return value


@app.get("/api/health")
def health_check():
    return {
        "status": "success",
        "message": "Recommendation API is running"
    }


@app.get("/api/products")
def get_products():
    products, similarity = load_model()

    names = (
        products["product_name"]
        .dropna()
        .drop_duplicates()
        .tolist()
    )

    return {
        "total_products": len(names),
        "products": names
    }


@app.post("/api/recommend")
def recommend_products(request: RecommendationRequest):

    products, similarity = load_model()

    product_name = request.product_name.strip()

    if request.top_n < 1 or request.top_n > 10:
        raise HTTPException(
            status_code=400,
            detail="top_n must be between 1 and 10"
        )

    matches = products[
        products["product_name"].str.lower()
        == product_name.lower()
    ]

    if matches.empty:
        raise HTTPException(
            status_code=404,
            detail="Product not found"
        )

    product_index = matches.index[0]

    scores = list(
        enumerate(similarity[product_index])
    )

    scores = sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )

    scores = [
        item for item in scores
        if item[0] != product_index
    ]

    scores = scores[:request.top_n]

    recommendations = []

    for index, score in scores:

        product = products.iloc[index]

        recommendations.append({
            "product_name": safe_value(product["product_name"]),
            "category": safe_value(product["category"]),
            "price": safe_value(product["discounted_price"]),
            "rating": safe_value(product["rating"]),
            "similarity_score": round(float(score) * 100, 2)
        })

    return {
        "selected_product": product_name,
        "recommendations": recommendations
    }
