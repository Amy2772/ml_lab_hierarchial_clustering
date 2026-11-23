
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, BisectingKMeans
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.decomposition import PCA

from scipy.cluster.hierarchy import linkage, dendrogram

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Hierarchical Clustering",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>
.stApp {
    background-color: #FFF5F7;
}
h1, h2, h3 {
    color: #9D4268;
}
.stMetric {
    background-color: #FFE4EC;
    padding: 15px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("📊 Hierarchical Clustering Dashboard")

st.markdown("""
### MAI511-2 – Advanced Machine Learning

**Lab Exercise 8: Demonstration of Hierarchical Clustering**

This application demonstrates Agglomerative and Divisive
Clustering using the Mall Customer Segmentation dataset.
""")

st.divider()

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("Mall_Customers.csv")
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("Mall_Customers.csv not found. Place it in the same folder as app.py.")
    st.stop()

# --------------------------------------------------
# DATASET INFORMATION
# --------------------------------------------------

st.header("1. Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Records", df.shape[0])
col2.metric("Total Attributes", df.shape[1])
col3.metric("Missing Values", int(df.isnull().sum().sum()))

st.subheader("Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

st.subheader("Dataset Information")

info_df = pd.DataFrame({
    "Attribute": df.columns,
    "Data Type": df.dtypes.astype(str).values,
    "Missing Values": df.isnull().sum().values
})

st.dataframe(info_df, use_container_width=True)

st.subheader("Statistical Description")
st.dataframe(df.describe(), use_container_width=True)

# --------------------------------------------------
# PREPROCESSING
# --------------------------------------------------

st.header("2. Data Preprocessing")

data = df[
    [
        "Age",
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]
].copy()

data = data.fillna(data.median())
data = data.drop_duplicates()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

st.success("Missing values handled and numerical features standardized.")

st.subheader("Standardized Data")
st.dataframe(
    pd.DataFrame(X_scaled, columns=data.columns).head(),
    use_container_width=True
)

# --------------------------------------------------
# DISTANCE MEASURES
# --------------------------------------------------

st.header("3. Distance Measures")

distance_metric = st.selectbox(
    "Select Distance Measure",
    ["euclidean", "manhattan", "cosine"]
)

distance_matrix = pairwise_distances(
    X_scaled,
    metric=distance_metric
)

st.write(f"Selected Distance: **{distance_metric.capitalize()}**")

fig, ax = plt.subplots(figsize=(8, 5))

sns.heatmap(
    distance_matrix[:20, :20],
    cmap="viridis",
    ax=ax
)

ax.set_title(f"{distance_metric.capitalize()} Distance Matrix")

st.pyplot(fig)
plt.close(fig)

# --------------------------------------------------
# AGGLOMERATIVE CLUSTERING
# --------------------------------------------------

st.header("4. Agglomerative Hierarchical Clustering")

col1, col2 = st.columns(2)

with col1:
    linkage_method = st.selectbox(
        "Select Linkage Method",
        ["single", "complete", "average", "ward"]
    )

with col2:
    n_clusters = st.slider(
        "Number of Clusters",
        min_value=2,
        max_value=10,
        value=5
    )

if linkage_method == "ward" and distance_metric != "euclidean":
    st.warning(
        "Ward linkage requires Euclidean distance. "
        "Euclidean distance will be used for Ward clustering."
    )
    selected_metric = "euclidean"
else:
    selected_metric = distance_metric

model = AgglomerativeClustering(
    n_clusters=n_clusters,
    linkage=linkage_method,
    metric=selected_metric
)

labels = model.fit_predict(X_scaled)

score = silhouette_score(
    X_scaled,
    labels,
    metric=selected_metric
)

col1, col2 = st.columns(2)

col1.metric("Number of Clusters", n_clusters)
col2.metric("Silhouette Score", round(score, 4))

# --------------------------------------------------
# CLUSTER DISTRIBUTION
# --------------------------------------------------

st.subheader("Cluster Distribution")

distribution = pd.Series(labels).value_counts().sort_index()

distribution_df = pd.DataFrame({
    "Cluster": distribution.index,
    "Customers": distribution.values
})

st.dataframe(distribution_df, use_container_width=True)

fig, ax = plt.subplots(figsize=(8, 4))

distribution.plot(
    kind="bar",
    ax=ax,
    color="#D9799B"
)

ax.set_title("Customer Distribution Across Clusters")
ax.set_xlabel("Cluster")
ax.set_ylabel("Number of Customers")

st.pyplot(fig)
plt.close(fig)

# --------------------------------------------------
# DENDROGRAM
# --------------------------------------------------

st.header("5. Dendrogram Visualization")

Z = linkage(
    X_scaled,
    method=linkage_method,
    metric=selected_metric
)

fig, ax = plt.subplots(figsize=(12, 5))

dendrogram(
    Z,
    truncate_mode="lastp",
    p=30,
    show_leaf_counts=True,
    ax=ax
)

ax.set_title(f"{linkage_method.capitalize()} Linkage Dendrogram")
ax.set_xlabel("Cluster Groups")
ax.set_ylabel("Distance")

st.pyplot(fig)
plt.close(fig)

# --------------------------------------------------
# DIVISIVE CLUSTERING
# --------------------------------------------------

st.header("6. Divisive Hierarchical Clustering")

div_model = BisectingKMeans(
    n_clusters=n_clusters,
    random_state=42
)

div_labels = div_model.fit_predict(X_scaled)

div_score = silhouette_score(
    X_scaled,
    div_labels
)

col1, col2 = st.columns(2)

col1.metric("Divisive Clusters", n_clusters)
col2.metric("Divisive Silhouette Score", round(div_score, 4))

st.subheader("Divisive Cluster Distribution")

st.bar_chart(
    pd.Series(div_labels).value_counts().sort_index()
)

# --------------------------------------------------
# PCA VISUALIZATION
# --------------------------------------------------

st.header("7. PCA Cluster Visualization")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

variance = sum(pca.explained_variance_ratio_) * 100

st.write(
    f"Variance explained by two principal components: "
    f"**{variance:.2f}%**"
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=labels,
    cmap="viridis",
    s=45
)

axes[0].set_title("Agglomerative Clustering")
axes[0].set_xlabel("Principal Component 1")
axes[0].set_ylabel("Principal Component 2")

axes[1].scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=div_labels,
    cmap="plasma",
    s=45
)

axes[1].set_title("Divisive Clustering")
axes[1].set_xlabel("Principal Component 1")
axes[1].set_ylabel("Principal Component 2")

plt.tight_layout()

st.pyplot(fig)
plt.close(fig)

# --------------------------------------------------
# COMPARISON
# --------------------------------------------------

st.header("8. Agglomerative vs Divisive Comparison")

comparison = pd.DataFrame({
    "Algorithm": [
        "Agglomerative",
        "Divisive"
    ],
    "Silhouette Score": [
        score,
        div_score
    ]
})

st.dataframe(comparison, use_container_width=True)

fig, ax = plt.subplots(figsize=(7, 4))

ax.bar(
    comparison["Algorithm"],
    comparison["Silhouette Score"],
    color=["#C76D91", "#8C9ED9"]
)

ax.set_ylabel("Silhouette Score")
ax.set_title("Clustering Algorithm Comparison")

st.pyplot(fig)
plt.close(fig)

# --------------------------------------------------
# CUSTOMER CLUSTER PROFILES
# --------------------------------------------------

st.header("9. Customer Cluster Interpretation")

cluster_data = data.copy()
cluster_data["Cluster"] = labels

cluster_summary = cluster_data.groupby("Cluster").mean()

st.subheader("Average Customer Characteristics")

st.dataframe(
    cluster_summary.round(2),
    use_container_width=True
)

st.subheader("Cluster Profile Visualization")

fig, ax = plt.subplots(figsize=(10, 5))

cluster_summary.plot(
    kind="bar",
    ax=ax
)

ax.set_title("Average Customer Characteristics by Cluster")
ax.set_ylabel("Average Value")
ax.set_xlabel("Cluster")

plt.xticks(rotation=0)
plt.tight_layout()

st.pyplot(fig)
plt.close(fig)

# --------------------------------------------------
# FINAL CONCLUSION
# --------------------------------------------------

st.header("10. Conclusion")

st.markdown("""
- Hierarchical clustering successfully groups customers
  based on their numerical characteristics.
- Agglomerative clustering follows a bottom-up approach.
- Bisecting K-Means demonstrates a divisive top-down approach.
- Different linkage methods and distance measures produce
  different cluster formations.
- PCA provides a two-dimensional visualization of the clusters.
- Silhouette scores help evaluate clustering quality.
""")

st.success("Hierarchical Clustering Analysis Completed Successfully!")

