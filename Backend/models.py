import numpy as np
import pandas as pd
import pickle
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 🔹 Load dataset
df = pd.read_csv("diabetes.csv")  # Ensure this file is uploaded

# 🔹 Step 1: Preprocessing (Handle missing values & scale data)
columns_to_fix = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[columns_to_fix] = df[columns_to_fix].replace(0, np.nan)

# Fill missing values with column mean
df.fillna(df.mean(), inplace=True)

# Define features
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
            'BMI', 'DiabetesPedigreeFunction', 'Age']

# Standardize features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[features])

# 🔹 Step 2: Train Clustering Models
models = {}

# K-Means
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["KMeans_Cluster"] = kmeans.fit_predict(scaled_data)
models["kmeans"] = kmeans

# Agglomerative Clustering
agglo = AgglomerativeClustering(n_clusters=3)
df["Agglo_Cluster"] = agglo.fit_predict(scaled_data)
models["agglo"] = agglo

# DBSCAN
dbscan = DBSCAN(eps=1.5, min_samples=5)
df["DBSCAN_Cluster"] = dbscan.fit_predict(scaled_data)
models["dbscan"] = dbscan

# KNN (Using K-Means labels as ground truth for training)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(scaled_data, df["KMeans_Cluster"])
df["KNN_Cluster"] = knn.predict(scaled_data)
models["knn"] = knn

# 🔹 Step 3: Evaluate Models
cluster_metrics = {}
for model_name, labels in {
    "K-Means": df["KMeans_Cluster"],
    "Agglomerative": df["Agglo_Cluster"],
    "DBSCAN": df["DBSCAN_Cluster"],
    "KNN": df["KNN_Cluster"],
}.items():
    unique_labels = set(labels)
    if len(unique_labels) < 2 or (len(unique_labels) == 2 and -1 in unique_labels):
        silhouette = None  # Silhouette score needs at least 2 clusters
    else:
        silhouette = silhouette_score(scaled_data, labels)
    
    cluster_metrics[model_name] = silhouette

# 🔹 Step 4: Save Models & Scaler
for name, model in models.items():
    with open(f"{name}_model.pkl", "wb") as f:
        pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# 🔹 Step 5: Save Silhouette Scores
with open("model_scores.pkl", "wb") as f:
    pickle.dump(cluster_metrics, f)

print("✅ Models, Scaler, and Scores saved successfully!")