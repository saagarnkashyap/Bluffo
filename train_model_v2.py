import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve
)
import json

# Load and preprocess the dataset
df = pd.read_csv("Shuffled_SAMPLE.csv")
df.columns = ["Sr.No", "text", "label"]
df.drop("Sr.No", axis=1, inplace=True)
df["label"] = df["label"].map({"REAL": 1, "FAKE": 0})

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

# TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Save model and vectorizer
joblib.dump(model, "fake_news_model_v3.joblib")
joblib.dump(vectorizer, "vectorizer_v3.joblib")
print("Model and vectorizer saved successfully.")

# Predict and evaluate
y_pred = model.predict(X_test_vec)

# Save classification report
report_dict = classification_report(y_test, y_pred, output_dict=True)
with open("classification_metrics_v3.json", "w") as f:
    json.dump(report_dict, f)

# Save confusion matrix image
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["FAKE", "REAL"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix_v3.png")
plt.close()

# Save ROC curve
fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test_vec)[:, 1])
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve_v3.png")
plt.close()

# Save Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_test, model.predict_proba(X_test_vec)[:, 1])
plt.figure()
plt.plot(recall, precision, color="green", lw=2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.tight_layout()
plt.savefig("precision_recall_v3.png")
plt.close()
