# Breast Cancer Survival Prediction using Machine and Deep Learning

This project aims to predict the **survival status** of breast cancer patients using a variety of machine learning and deep learning models. The dataset is derived from the [SEER Program (Surveillance, Epidemiology, and End Results)](https://seer.cancer.gov/) and contains clinical and demographic features of patients diagnosed with breast cancer between 2006 and 2010.

> **Course Project**  
> *Yapay Zekaya Giriş (Introduction to Artificial Intelligence)*

---

## 📌 Project Goals

- Perform Exploratory Data Analysis (EDA) on the SEER Breast Cancer dataset.
- Handle missing and outlier values effectively.
- Apply at least **5 machine learning models** and **3 deep learning models**.
- Evaluate models using **Accuracy**, **F1 Score**, **Precision**, **Recall**, **AUC (ROC)**, and **Cohen's Kappa**.
- Identify the most important features affecting survival.
- Select and save the best-performing model.
- Build a function to predict the survival status of new patients based on input features.

---

## 🗂️ Dataset Overview

- 📁 Source: SEER Breast Cancer Dataset (2006–2010)  
- 🧪 Samples: 4024 patients  
- 🔬 Features include:
  - Age, Race, Marital Status
  - Tumor Stage (T, N, 6th AJCC Stage)
  - Tumor Size, Grade
  - Hormone Receptor Status (Estrogen, Progesterone)
  - Number of lymph nodes examined and found positive
  - Survival Months
  - Survival Status (Alive / Dead)

---

## 🧠 Models Implemented

### 🔹 Machine Learning
- Logistic Regression
- Random Forest
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Decision Tree
- XGBoost

### 🔹 Deep Learning
- Feedforward Neural Network (FNN)
- Multi-layer Perceptron (MLP)
- Deep Neural Network (DNN)
- Convolutional Neural Network (CNN)

Each model includes:
- Hyperparameter tuning (e.g., GridSearchCV)
- Feature selection (e.g., RFE)
- Evaluation using multiple metrics and ROC curves

---

## 📊 Feature Engineering

- Encoded categorical variables into numeric form.
- Introduced a new engineered feature: `Risk Group` based on tumor stage and grade.
- Feature selection applied using Recursive Feature Elimination (RFE).

---

## 📈 Evaluation Metrics

| Metric            | Description                                  |
|-------------------|----------------------------------------------|
| Accuracy          | Overall correctness of the model             |
| Precision         | Correctly predicted positives / Total predicted positives |
| Recall (Sensitivity) | Correctly predicted positives / All actual positives |
| F1 Score          | Harmonic mean of precision and recall        |
| ROC-AUC           | Area under the ROC Curve                     |
| Cohen’s Kappa     | Inter-rater agreement between predicted and true classes |

---

## 📌 Best Model Selection

After evaluating all models, the one with the best **Accuracy and F1 Score** is automatically selected and saved as `best_model.pkl`.

---

## 🔮 Predicting on New Data

The project includes a ready-to-use function to:
- Collect new patient input interactively
- Encode and scale the inputs
- Use the trained best model to predict survival outcome


## 🔎 Example: Predicting a New Patient

To predict the survival status for a new patient, you can run the following interactive session built into the script:

```python
# Step 1: Collect new input from the user
new_sample = get_input_for_features()

# Step 2: Scale the input using the same scaler used during training
new_sample_scaled = scaler.transform(new_sample)

# Step 3: Predict using the best model
prediction = predict_new_samples(best_model, new_sample_scaled)

# Step 4: Output result
print("✅ Patient is likely to SURVIVE." if prediction[0] == 1 else "❌ Patient is likely to NOT survive.")
