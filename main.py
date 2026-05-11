import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import tkinter as tk
from tkinter import ttk, messagebox

# ---------------------------------------
# LOAD DATASET
# ---------------------------------------

data = pd.read_csv("dataset.csv")

# ---------------------------------------
# REMOVE DUPLICATES
# ---------------------------------------

data = data.drop_duplicates()

# ---------------------------------------
# HANDLE MISSING VALUES
# ---------------------------------------

data = data.fillna("")

# ---------------------------------------
# GET SYMPTOM COLUMNS
# ---------------------------------------

symptom_columns = data.columns[:-1]

# ---------------------------------------
# CONVERT SYMPTOMS INTO LIST
# ---------------------------------------

symptoms = data[symptom_columns].values.tolist()

# Remove empty values
symptoms = [
    [symptom for symptom in row if symptom != ""]
    for row in symptoms
]

# ---------------------------------------
# MULTILABEL BINARIZER
# ---------------------------------------

mlb = MultiLabelBinarizer()

X = mlb.fit_transform(symptoms)

# ---------------------------------------
# TARGET COLUMN
# ---------------------------------------

y = data["Disease"]

# ---------------------------------------
# TRAIN TEST SPLIT
# ---------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# ---------------------------------------
# RANDOM FOREST MODEL
# ---------------------------------------

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    random_state=42
)

model.fit(X_train, y_train)

# ---------------------------------------
# MODEL EVALUATION
# ---------------------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n================================")
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("================================")

# ---------------------------------------
# GET ONLY SYMPTOMS
# ---------------------------------------

all_symptoms = set()

for col in symptom_columns:

    values = data[col].unique()

    for value in values:

        value = str(value).strip().lower()

        if value != "":
            all_symptoms.add(value)

# REMOVE DISEASE NAMES
diseases = set(data["Disease"].str.lower().unique())

all_symptoms = all_symptoms - diseases

# SORT SYMPTOMS
all_symptoms = sorted(list(all_symptoms))

# ---------------------------------------
# GUI WINDOW
# ---------------------------------------

root = tk.Tk()

root.title("Disease Prediction System")

root.geometry("900x800")

root.configure(bg="#EAF2F8")

# ---------------------------------------
# TITLE
# ---------------------------------------

title = tk.Label(
    root,
    text="Disease Prediction System",
    font=("Arial", 28, "bold"),
    bg="#EAF2F8",
    fg="#1B4F72"
)

title.pack(pady=20)

# ---------------------------------------
# ACCURACY LABEL
# ---------------------------------------

accuracy_label = tk.Label(
    root,
    text=f"Model Accuracy: {accuracy * 100:.2f}%",
    font=("Arial", 15, "bold"),
    bg="#EAF2F8",
    fg="green"
)

accuracy_label.pack(pady=10)

# ---------------------------------------
# CREATE DROPDOWNS
# ---------------------------------------

entries = []

for i in range(4):

    frame = tk.Frame(root, bg="#EAF2F8")

    frame.pack(pady=12)

    label = tk.Label(
        frame,
        text=f"Symptom {i+1}",
        font=("Arial", 14, "bold"),
        width=15,
        anchor="w",
        bg="#EAF2F8"
    )

    label.pack(side=tk.LEFT, padx=10)

    combo = ttk.Combobox(
        frame,
        values=all_symptoms,
        width=40,
        font=("Arial", 12)
    )

    combo.pack(side=tk.LEFT)

    entries.append(combo)

# ---------------------------------------
# RESULT LABEL
# ---------------------------------------

result_label = tk.Label(
    root,
    text="",
    font=("Arial", 17, "bold"),
    bg="#EAF2F8",
    justify="left"
)

result_label.pack(pady=25)

# ---------------------------------------
# PREDICTION FUNCTION
# ---------------------------------------

def predict_disease():

    try:

        user_symptoms = []

        for entry in entries:

            symptom = entry.get().strip().lower()

            if symptom != "":
                user_symptoms.append(symptom)

        # No symptoms selected
        if len(user_symptoms) == 0:

            messagebox.showwarning(
                "Input Error",
                "Please select at least one symptom"
            )

            return

        # ---------------------------------------
        # TRANSFORM INPUT
        # ---------------------------------------

        input_data = mlb.transform([user_symptoms])

        # ---------------------------------------
        # PREDICT PROBABILITIES
        # ---------------------------------------

        probabilities = model.predict_proba(input_data)[0]

        classes = model.classes_

        # ---------------------------------------
        # GET TOP 3 DISEASES
        # ---------------------------------------

        top_indices = np.argsort(probabilities)[-3:][::-1]

        result_text = "Top Possible Diseases:\n\n"

        for index in top_indices:

            disease = classes[index]

            confidence = probabilities[index] * 100

            result_text += (
                f"{disease}  →  {confidence:.2f}%\n"
            )

        # ---------------------------------------
        # DISPLAY RESULT
        # ---------------------------------------

        result_label.config(
            text=result_text,
            fg="green"
        )

    except Exception as e:

        messagebox.showerror("Error", str(e))

# ---------------------------------------
# PREDICT BUTTON
# ---------------------------------------

predict_button = tk.Button(
    root,
    text="Predict Disease",
    font=("Arial", 15, "bold"),
    bg="#1ABC9C",
    fg="white",
    padx=20,
    pady=10,
    command=predict_disease
)

predict_button.pack(pady=30)

# ---------------------------------------
# FOOTER
# ---------------------------------------

footer = tk.Label(
    root,
    text="Healthcare Analytics using Machine Learning",
    font=("Arial", 11, "italic"),
    bg="#EAF2F8",
    fg="#566573"
)

footer.pack(side=tk.BOTTOM, pady=20)

# ---------------------------------------
# RUN APPLICATION
# ---------------------------------------

root.mainloop()
