Heart Disease Prediction with Machine Learning
📌 Overview

This project predicts the presence of heart disease using three machine learning models:

Decision Tree
Logistic Regression
Naive Bayes (GaussianNB)

The dataset (heart.csv) contains patient information such as age, sex, cholesterol, blood pressure, and other heart-related features. The models are evaluated using accuracy, precision, recall, F1-score, and confusion matrices, with additional cross-validation for robust performance measurement.

🛠️ Dataset Features
Age – Patient age in years
Sex – Male/Female
ChestPainType – Type of chest pain (ATA, NAP, ASY, TA)
RestingBP – Resting blood pressure
Cholesterol – Serum cholesterol in mg/dl
FastingBS – Fasting blood sugar > 120 mg/dl
RestingECG – Resting electrocardiogram results
MaxHR – Maximum heart rate achieved
ExerciseAngina – Exercise-induced angina (Y/N)
Oldpeak – ST depression induced by exercise
ST_Slope – Slope of the peak exercise ST segment
HeartDisease – Target variable (0 = No disease, 1 = Disease)

Categorical variables are one-hot encoded for modeling.

⚙️ Models Implemented
Decision Tree (max_depth=2)
Simple and interpretable.
Slightly lower accuracy due to shallow depth.
Logistic Regression
Linear classifier.
High precision, good for minimizing false positives.
Naive Bayes (GaussianNB)
Probabilistic classifier assuming feature independence.
Strong recall, good at catching actual heart disease cases.

⚙️ Models Implemented
Decision Tree (max_depth=2)
Simple and interpretable.
Slightly lower accuracy due to shallow depth.
Logistic Regression
Linear classifier.
High precision, good for minimizing false positives.
Naive Bayes (GaussianNB)
Probabilistic classifier assuming feature independence.
Strong recall, good at catching actual heart disease cases.

📊 Evaluation Metrics
Accuracy – Correct predictions / total predictions
Precision – True positives / predicted positives
Recall – True positives / actual positives
F1-score – Harmonic mean of precision and recall
Confusion Matrix – Visualization of predictions vs actuals

Example Results (Test Set)

| Model               | Accuracy | Precision | Recall | F1-score |
| ------------------- | -------- | --------- | ------ | -------- |
| Decision Tree       | 0.808    | 0.825     | 0.860  | 0.842    |
| Logistic Regression | 0.877    | 0.922     | 0.866  | 0.893    |
| Naive Bayes         | 0.877    | 0.917     | 0.872  | 0.894    |

Cross-Validation (5-Fold)
| Model               | CV Accuracy Mean | CV Accuracy Std |
| ------------------- | ---------------- | --------------- |
| Decision Tree       | 0.790            | 0.058           |
| Logistic Regression | 0.833            | 0.035           |
| Naive Bayes         | 0.838            | 0.043           |

