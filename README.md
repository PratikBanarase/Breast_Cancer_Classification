🩺 Breast Cancer Classification Project
📘 Overview

This project aims to build a machine learning model that classifies breast tumors as Malignant (Cancerous) or Benign (Non-cancerous) based on cell nuclei features derived from digitized images of breast mass.
By leveraging supervised learning algorithms, this project contributes toward early cancer detection and diagnosis support.

📊 Dataset

Dataset Name: Breast Cancer Wisconsin (Diagnostic) Dataset

Source:

UCI Machine Learning Repository

or via sklearn.datasets.load_breast_cancer()

🧾 Features:

30 numerical input features such as:

radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, etc.

Target Variable:

0 → Malignant

1 → Benign

⚙️ Technologies Used

Programming Language: Python

Libraries:

numpy

pandas

matplotlib

seaborn

scikit-learn

joblib

streamlit (optional, for web app)

🧠 Model Building Steps
1️⃣ Data Preprocessing

Handled missing or irrelevant data (if any)

Standardized numerical features using StandardScaler

Split data into training (80%) and testing (20%) sets

2️⃣ Model Training

Tested multiple classification models:

Logistic Regression

Support Vector Machine (SVM)

Random Forest Classifier

K-Nearest Neighbors (KNN)

Chose the best-performing model using GridSearchCV for hyperparameter tuning

3️⃣ Model Evaluation

Metrics used:

Accuracy

Precision

Recall

F1-score

Confusion Matrix

Best model saved using joblib as breast_cancer_model.pkl

📈 Results

The final model achieved:

Accuracy: ≈ 98%

High recall for both malignant and benign predictions

Model demonstrated strong reliability for early cancer detection
