#!/usr/bin/env python
# coding: utf-8

# ============================================================
# DISEASE PREDICTION SYSTEM
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.decomposition import PCA

# ============================================================
# CREATE FOLDER FOR SAVED GRAPHS
# ============================================================

os.makedirs("graphs", exist_ok=True)

# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv("Testing.csv")

print(f"\nDataset Shape: {df.shape}\n")

print(df.head())

# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

df.columns = [col.strip() for col in df.columns]

# ============================================================
# CHECK MISSING VALUES
# ============================================================

print("\nMissing Values:\n")

print(df.isnull().sum())

# ============================================================
# REMOVE DUPLICATES
# ============================================================

df = df.drop_duplicates()

# ============================================================
# HANDLE MISSING VALUES
# ============================================================

df = df.fillna(0)

# ============================================================
# ENCODE TARGET VARIABLE
# ============================================================

le = LabelEncoder()

df['prognosis'] = le.fit_transform(df['prognosis'])

# Disease mapping
disease_mapping = dict(
    zip(le.transform(le.classes_), le.classes_)
)

# ============================================================
# EDA - TOP SYMPTOMS
# ============================================================

plt.figure(figsize=(12, 6))

symptom_counts = (
    df.iloc[:, :-1]
    .sum()
    .sort_values(ascending=False)
    .head(15)
)

sns.barplot(
    x=symptom_counts.values,
    y=symptom_counts.index,
    palette='magma'
)

plt.title('Top 15 Most Common Symptoms in Dataset')

plt.xlabel('Frequency')

# SAVE GRAPH
plt.savefig("graphs/top_symptoms.png")

plt.show()

# ============================================================
# FEATURES AND TARGET
# ============================================================

X = df.drop('prognosis', axis=1)

y = df['prognosis']

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ============================================================
# MODEL TRAINING
# ============================================================

models = {

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42
    ),

    "SVM": SVC(
        kernel='linear',
        probability=True
    ),

    "Naive Bayes": GaussianNB()
}

# ============================================================
# TRAIN & EVALUATE MODELS
# ============================================================

print("\n================ MODEL ACCURACY ================\n")

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    acc = accuracy_score(y_test, predictions)

    print(f"{name} Accuracy: {acc*100:.2f}%")

# ============================================================
# RANDOM FOREST MODEL
# ============================================================

rf_model = models["Random Forest"]

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importances = rf_model.feature_importances_

indices = np.argsort(importances)[-10:]

plt.figure(figsize=(10, 5))

plt.title('Top 10 Important Symptoms')

plt.barh(
    range(len(indices)),
    importances[indices],
    color='skyblue',
    align='center'
)

plt.yticks(
    range(len(indices)),
    [X.columns[i] for i in indices]
)

plt.xlabel('Importance Score')

# SAVE GRAPH
plt.savefig("graphs/feature_importance.png")

plt.show()

# ============================================================
# DISEASE PREDICTION FUNCTION
# ============================================================

def predict_disease_from_symptoms(user_symptoms):

    input_vector = np.zeros(len(X.columns))

    for symptom in user_symptoms:

        if symptom in X.columns:

            idx = X.columns.get_loc(symptom)

            input_vector[idx] = 1

        else:

            print(f"Warning: Symptom '{symptom}' not found.")

    # Prediction probabilities
    probabilities = rf_model.predict_proba([input_vector])[0]

    classes = rf_model.classes_

    # Top 3 diseases
    top_indices = np.argsort(probabilities)[-3:][::-1]

    print("\n====================================")
    print("Top Possible Diseases:\n")

    for idx in top_indices:

        disease = disease_mapping[classes[idx]]

        confidence = probabilities[idx] * 100

        print(f"{disease} → {confidence:.2f}%")

    print("====================================")

    prediction_idx = top_indices[0]

    return disease_mapping[classes[prediction_idx]]

# ============================================================
# USER INPUT
# ============================================================

print("\nAvailable Symptoms:\n")

for symptom in X.columns:
    print(symptom)

print("\nEnter Symptoms:\n")

symptoms_input = input(
    "Enter symptoms separated by comma: "
)

test_symptoms = [
    symptom.strip()
    for symptom in symptoms_input.split(",")
]

# ============================================================
# PREDICTION
# ============================================================

result = predict_disease_from_symptoms(test_symptoms)

print(f"\nFinal Predicted Disease: {result}")

# ============================================================
# CONFUSION MATRIX
# ============================================================

plt.figure(figsize=(12, 10))

sns.heatmap(
    confusion_matrix(
        y_test,
        rf_model.predict(X_test)
    ),
    annot=True,
    cmap='Greens'
)

plt.title('Confusion Matrix')

plt.xlabel('Predicted')

plt.ylabel('Actual')

# SAVE GRAPH
plt.savefig("graphs/confusion_matrix.png")

plt.show()

# ============================================================
# SYMPTOM FREQUENCY GRAPH
# ============================================================

symptom_frequencies = (
    X.sum()
    .sort_values(ascending=False)
    .head(15)
)

fig, ax = plt.subplots(figsize=(14, 6))

sns.barplot(
    x=symptom_frequencies.index,
    y=symptom_frequencies.values,
    palette="crest",
    ax=ax,
    edgecolor="black"
)

ax.set_title(
    "Top 15 Most Prevalent Symptoms",
    fontsize=16,
    weight='bold'
)

ax.set_ylabel("Frequency")

ax.set_xlabel("Symptoms")

plt.xticks(rotation=45)

plt.tight_layout()

# SAVE GRAPH
plt.savefig("graphs/symptom_frequency.png")

plt.show()

# ============================================================
# CORRELATION HEATMAP
# ============================================================

top_symptom_cols = symptom_frequencies.index.tolist()

correlation_matrix = X[top_symptom_cols].corr()

fig, ax = plt.subplots(figsize=(12, 10))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=.5,
    ax=ax
)

ax.set_title("Symptom Correlation Heatmap")

plt.tight_layout()

# SAVE GRAPH
plt.savefig("graphs/correlation_heatmap.png")

plt.show()

# ============================================================
# PCA ANALYSIS
# ============================================================

symptom_cols = [
    c for c in df.columns
    if c != 'prognosis'
]

X_pca_input = df[symptom_cols].values

y_raw = df['prognosis']

# Standardize Data
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_pca_input)

# PCA
pca = PCA(n_components=2, random_state=42)

X_pca = pca.fit_transform(X_scaled)

# PCA DataFrame
pca_dataframe = pd.DataFrame(
    data=X_pca,
    columns=[
        'Principal Component 1',
        'Principal Component 2'
    ]
)

pca_dataframe['Disease Diagnosis'] = y_raw.values

# ============================================================
# PCA SCATTER PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(14, 10))

sns.scatterplot(
    x='Principal Component 1',
    y='Principal Component 2',
    hue='Disease Diagnosis',
    data=pca_dataframe,
    ax=ax,
    palette='tab20',
    s=100
)

ax.set_title(
    "PCA Disease Cluster Visualization",
    fontsize=16,
    weight='bold'
)

plt.tight_layout()

# SAVE GRAPH
plt.savefig("graphs/pca_scatter.png")

plt.show()

# ============================================================
# PCA LOADINGS
# ============================================================

loadings = pd.DataFrame(
    pca.components_.T,
    columns=['PC1', 'PC2'],
    index=symptom_cols
)

plt.figure(figsize=(12, 6))

top_pc1 = (
    loadings['PC1']
    .sort_values(ascending=False)
    .head(15)
)

sns.barplot(
    x=top_pc1.values,
    y=top_pc1.index,
    palette='coolwarm'
)

plt.title(
    'Top PCA Symptom Contributions'
)

plt.xlabel('Loading Value')

# SAVE GRAPH
plt.savefig("graphs/pca_loadings.png")

plt.show()

# ============================================================
# PCA EXPLAINED VARIANCE
# ============================================================

pca_full = PCA().fit(X_scaled)

plt.figure(figsize=(10, 5))

plt.plot(
    np.cumsum(
        pca_full.explained_variance_ratio_
    ),
    marker='o',
    color='darkred'
)

plt.title(
    'PCA Cumulative Explained Variance',
    fontsize=15,
    weight='bold'
)

plt.xlabel('Number of Components')

plt.ylabel('Variance Explained')

plt.grid(True, alpha=0.3)

# SAVE GRAPH
plt.savefig("graphs/pca_variance.png")

plt.show()

# ============================================================
# SAVE RESULTS
# ============================================================

with open("prediction_result.txt", "w") as file:

    file.write(
        f"Final Predicted Disease: {result}\n"
    )

print("\nResults saved successfully.")

print("\nGraphs saved inside 'graphs' folder.")