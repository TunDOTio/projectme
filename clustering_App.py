import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
import pickle

st.title("🧠 ML Playground: Classification, Regression, Clustering")

# Load Telco data + model
@st.cache_resource
def load_telco_model():
    with open("telco_model_data.pkl", "rb") as f:
        return pickle.load(f)

telco = load_telco_model()

# Tabs for modes
tabs = st.tabs(["🧪 Classification", "📈 Regression", "🔍 Clustering"])

# ---------------- Classification ----------------
with tabs[0]:
    st.header("Classification: Telco Customer Churn")
    y_true = pd.Series(telco['y_test'])
    y_pred = pd.Series(telco['y_pred'])
    st.write(f"**Accuracy:** {accuracy_score(y_true, y_pred):.2f}")
    st.write("### Confusion Matrix")
    st.write(pd.crosstab(y_true, y_pred, rownames=["Actual"], colnames=["Predicted"]))
    st.write("### Sample Predictions")
    st.write(pd.DataFrame({"Actual": y_true[:10], "Predicted": y_pred[:10]}))


# ---------------- Regression ----------------
with tabs[1]:
    st.header("Regression: Predict Total Charges")
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = df[df["TotalCharges"] != " "]  # Remove blanks
    df["TotalCharges"] = df["TotalCharges"].astype(float)
    df["tenure"] = df["tenure"].astype(float)

    X = df[["tenure"]]
    y = df["TotalCharges"]
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    st.write(f"**MSE:** {mean_squared_error(y, y_pred):.2f}")
    fig, ax = plt.subplots()
    ax.scatter(X, y, alpha=0.3, label="Actual")
    ax.plot(X, y_pred, color="red", label="Predicted")
    ax.set_xlabel("Tenure")
    ax.set_ylabel("Total Charges")
    ax.set_title("Linear Regression: Tenure vs Total Charges")
    ax.legend()
    st.pyplot(fig)

# ---------------- Clustering ----------------
with tabs[2]:
    st.header("K-Means Clustering: Telco Data")

    # Load data from pickle
    X_telco = telco["X_train"]  # Already scaled
    k = st.slider("Select number of clusters (k)", 2, 10, 3)

    # Apply K-Means
    kmeans_telco = KMeans(n_clusters=k, random_state=42)
    labels_telco = kmeans_telco.fit_predict(X_telco)

    # Dimensionality reduction for visualization
    pca_telco = PCA(n_components=2)
    reduced_telco = pca_telco.fit_transform(X_telco)
    reduced_df_telco = pd.DataFrame(reduced_telco, columns=["PCA1", "PCA2"])
    reduced_df_telco["Cluster"] = labels_telco

    # Plot clusters
    fig_telco, ax_telco = plt.subplots()
    for cluster in range(k):
        cluster_data = reduced_df_telco[reduced_df_telco["Cluster"] == cluster]
        ax_telco.scatter(cluster_data["PCA1"], cluster_data["PCA2"], label=f"Cluster {cluster}")
    ax_telco.set_title("K-Means Clusters (Telco Data, 2D PCA)")
    ax_telco.set_xlabel("PCA1")
    ax_telco.set_ylabel("PCA2")
    ax_telco.legend()
    st.pyplot(fig_telco)

    st.dataframe(reduced_df_telco.head(10))