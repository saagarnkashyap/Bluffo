import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve,
    auc
)
import json

# Load dataset
df = pd.read_csv("Shuffled_SAMPLE.csv")
df.columns = ["Sr.No", "text", "label"]
df.drop("Sr.No", axis=1, inplace=True)
df["label"] = df["label"].map({"REAL": 1, "FAKE": 0})

# Split
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

# Vectorize
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train_vec, y_train)

# Save model and vectorizer
joblib.dump(model, "logreg_model.pkl")
joblib.dump(vectorizer, "logreg_vectorizer.pkl")
print("Model and vectorizer saved successfully.")

# Evaluate
y_pred = model.predict(X_test_vec)

# Save classification report
report = classification_report(y_test, y_pred, output_dict=True)
with open("classification_metrics_v3.json", "w") as f:
    json.dump(report, f)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["FAKE", "REAL"])
disp.plot(cmap="Purples")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix_v3.png")
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test_vec)[:, 1])
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve_v3.png")
plt.close()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, model.predict_proba(X_test_vec)[:, 1])
plt.figure()
plt.plot(recall, precision, lw=2, color="green")
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.tight_layout()
plt.savefig("precision_recall_v3.png")
plt.close()
