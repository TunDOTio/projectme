# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 22:14:00 2025

@author: Nongnuch
"""

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
    y_true = telco['y_test']
    y_pred = telco['y_pred']
    st.write(f"**Accuracy:** {accuracy_score(y_true, y_pred):.2f}")
    st.write("### Confusion Matrix")
    st.write(pd.crosstab(y_true, y_pred, rownames=["Actual"], colnames=["Predicted"]))
    st.write("### Sample Predictions")
    st.write(pd.DataFrame({"Actual": y_true.values[:10], "Predicted": y_pred[:10]}))

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
    st.header("K-Means Clustering: Iris Dataset")
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)

    k = st.slider("Select number of clusters (k)", 2, 10, 3)
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X)
    reduced_df = pd.DataFrame(reduced, columns=["PCA1", "PCA2"])
    reduced_df["Cluster"] = labels

    fig, ax = plt.subplots()
    for cluster in range(k):
        cluster_data = reduced_df[reduced_df["Cluster"] == cluster]
        ax.scatter(cluster_data["PCA1"], cluster_data["PCA2"], label=f"Cluster {cluster}")
    ax.set_title("Clusters (2D PCA Projection)")
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")
    ax.legend()
    st.pyplot(fig)
    st.dataframe(reduced_df.head(10))
