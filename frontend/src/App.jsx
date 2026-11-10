
import React, { useEffect, useState } from "react";

function App() {
  const [products, setProducts] = useState([]);
  const [selectedProduct, setSelectedProduct] = useState("");
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch("/api/products")
      .then((response) => {
        if (!response.ok) throw new Error("API error");
        return response.json();
      })
      .then((data) => setProducts(data.products))
      .catch(() => setError("Products load nahi ho paaye."));
  }, []);

  const getRecommendations = async () => {
    if (!selectedProduct) {
      setError("Please select a product.");
      return;
    }

    setLoading(true);
    setError("");
    setRecommendations([]);

    try {
      const response = await fetch("/api/recommend", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          product_name: selectedProduct,
          top_n: 5
        })
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Something went wrong");
      }

      setRecommendations(data.recommendations);
    } catch (err) {
      setError(err.message);
    }

    setLoading(false);
  };

  return (
    <div className="app">
      <header>
        <h1>ElectroFind</h1>
        <p>Smart Electronics Product Recommendation System</p>
      </header>

      <main>
        <h2>Find Similar Products</h2>
        <p>Select a product to discover similar products.</p>

        <select
          value={selectedProduct}
          onChange={(e) => setSelectedProduct(e.target.value)}
        >
          <option value="">Select a Product</option>

          {products.map((product, index) => (
            <option key={index} value={product}>
              {product}
            </option>
          ))}
        </select>

        <button onClick={getRecommendations} disabled={loading}>
          {loading ? "Finding..." : "Get Recommendations"}
        </button>

        {error && <p className="error">{error}</p>}

        <section className="results">
          {recommendations.map((item, index) => (
            <div className="card" key={index}>
              <h3>{item.product_name}</h3>
              <p>Category: {item.category}</p>
              <p>Price: {item.price}</p>
              <p>Rating: {item.rating}</p>
              <p>Similarity: {item.similarity_score}%</p>
            </div>
          ))}
        </section>
      </main>
    </div>
  );
}

export default App;
