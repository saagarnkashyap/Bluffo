import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

# Load the dataset
df = pd.read_csv("Shuffled_SAMPLE.csv")

# Rename columns to standard names
df.columns = ["Sr.No", "text", "label"]
df.drop("Sr.No", axis=1, inplace=True)

# Encode labels
df["label"] = df["label"].map({"REAL": 1, "FAKE": 0})

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

# Vectorize text
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model with balanced class weights
model = PassiveAggressiveClassifier(max_iter=1000, class_weight="balanced")
model.fit(X_train_vec, y_train)

# Save model and vectorizer
with open("fake_news_model_v2.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer_v2.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model and vectorizer saved successfully.")

# Predictions
y_pred = model.predict(X_test_vec)

# Save classification report
report = classification_report(y_test, y_pred)
with open("classification_metrics_v2.txt", "w") as f:
    f.write(report)

# Save confusion matrix image
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["FAKE", "REAL"])
disp.plot(cmap="Blues")  # Try "Blues", "Purples", "Greens", etc.
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix_v2.png")
plt.close()

# Save ROC curve
fpr, tpr, _ = roc_curve(y_test, model.decision_function(X_test_vec))
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve_v2.png")
plt.close()

print("Training complete. Performance reports saved.")



from sklearn.metrics import precision_recall_curve
import json

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, model.decision_function(X_test_vec))
plt.figure()
plt.plot(recall, precision, color="green", lw=2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.grid(True)
plt.tight_layout()
plt.savefig("precision_recall_v2.png")
plt.close()

# Save classification report as JSON for table view
report_dict = classification_report(y_test, y_pred, output_dict=True)
with open("classification_metrics_v2.json", "w") as f:
    json.dump(report_dict, f)

