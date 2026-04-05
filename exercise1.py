import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import numpy as np

# Load and preprocess data
df = pd.read_csv("heart.csv")
df = pd.get_dummies(df, drop_first=True)
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# =========================
# 🌳 Decision Tree
# =========================
dt_model = DecisionTreeClassifier(max_depth=2, random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

# =========================
# 📈 Logistic Regression
# =========================
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

# =========================
# 🧮 Naive Bayes
# =========================
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)

# =========================
# Evaluation on test set
# =========================
models = {
    "Decision Tree": dt_pred,
    "Logistic Regression": lr_pred,
    "Naive Bayes": nb_pred
}

results = []

for name, pred in models.items():
    acc = accuracy_score(y_test, pred)
    report = classification_report(y_test, pred, output_dict=True)
    precision = report['1']['precision']
    recall = report['1']['recall']
    f1 = report['1']['f1-score']
    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    })

results_df = pd.DataFrame(results)
print("\n✅ Model Comparison Table")
print(results_df)

# =========================
# Confusion Matrices
# =========================
for name, pred in models.items():
    print(f"\nConfusion Matrix: {name}")
    disp = ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["No Disease", "Disease"])
    plt.title(f"Confusion Matrix: {name}")
    plt.show()

# =========================
# Accuracy Bar Chart
# =========================
plt.figure(figsize=(8,5))
plt.bar(results_df['Model'], results_df['Accuracy'], color=['skyblue','orange','green'])
plt.ylim(0,1)
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.show()

# =========================
# Cross-Validation
# =========================
cv_results = []
cv_models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=2, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": GaussianNB()
}



for name, model in cv_models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    cv_results.append({
        "Model": name,
        "CV Accuracy Mean": np.mean(scores),
        "CV Accuracy Std": np.std(scores)
    })

cv_results_df = pd.DataFrame(cv_results)
print("\n✅ Cross-Validation Results (5-Fold)")
print(cv_results_df)