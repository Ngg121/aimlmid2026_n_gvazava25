import os
import re
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score


DATASET_PATH = "/Users/ninogvazava/Desktop/2. Dataset.csv"


EMAIL_TEXT = """
Hi Alex,

Thanks for today’s meeting. I’m sharing a quick summary and the agreed next steps.

Summary:
• We reviewed the Q1 project timeline
• Confirmed resource allocation for the development team
• Agreed to finalize the budget by January 15

Next Steps:
• I will send the updated project plan by Friday
• You will review and provide feedback early next week
• We will reconvene on January 17 at 11:00 AM

Please let me know if I missed anything or if you’d like to add comments.

Best regards,

Nino
Project Manager
"""

# =========================
# 3) SPAM WORD LIST (FOR  PARSER)
# =========================
SPAM_WORDS = {
    "free", "winner", "win", "prize", "cash", "credit", "loan", "offer",
    "urgent", "limited", "deal", "discount", "bonus", "click", "buy",
    "cheap", "money", "guarantee", "congratulations"
}

URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
WORD_RE = re.compile(r"[A-Za-z0-9']+")

# =========================
# 4) PARSER (MATCHES 4-NUMBER OUTPUT)
#    words, links, capital_words, spam_word_count
# =========================
def parse_email(text: str):
    tokens = WORD_RE.findall(text)
    words = len(tokens)
    links = len(URL_RE.findall(text))
    capital_words = sum(1 for t in tokens if len(t) >= 2 and t.isupper())
    spam_word_count = sum(1 for t in tokens if t.lower() in SPAM_WORDS)
    return words, links, capital_words, spam_word_count

# =========================
# 5) LOAD DATASET (ROBUST)
#    - Assumes last column is label (spam/legitimate or 0/1)
#    - Uses first 4 columns as features (must match parser order!)
# =========================
def load_dataset(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")

    df = pd.read_csv(path)

    # If  CSV has more columns than needed, we keep first 4 as X
    # and assume last column is the label y.
    X = df.iloc[:, :4]
    y = df.iloc[:, -1]

    return X, y

# =========================
# 6) MAIN
# =========================
if __name__ == "__main__":
    # ---- Load data
    X, y = load_dataset(DATASET_PATH)

    # ---- Split 70/30
    stratify_arg = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.30,
        random_state=42,
        stratify=stratify_arg
    )

    # ---- Train Logistic Regression
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    # =========================
    # 2) VALIDATION (30%)
    # =========================
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

    print("\n=== Validation on 30% Test Split ===")
    print("Classes (order used in confusion matrix):", list(model.classes_))
    print("Accuracy:", round(acc, 4))
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(cm)

    # =========================
    # 3) PREDICT FOR ONE EMAIL
    # =========================
    w, l, c, s = parse_email(EMAIL_TEXT)

    # Make sure feature columns align with dataset columns
    email_features = pd.DataFrame([[w, l, c, s]], columns=X.columns)

    email_pred = model.predict(email_features)[0]
    print("\n=== Single Email Prediction ===")
    print("Extracted features:", (w, l, c, s))
    print("Predicted label:", email_pred)

    # Probability output (if available)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(email_features)[0]
        print("\nClass probabilities:")
        for cls, p in zip(model.classes_, proba):
            print(f"  {cls}: {p:.4f}")


import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# =========================
# PATHS
# =========================
DATASET_PATH = "/Users/ninogvazava/Desktop/2. Dataset.csv"
OUTPUT_DIR = "/Users/ninogvazava/Desktop"

CSV_OUTPUT = os.path.join(OUTPUT_DIR, "confusion_matrix.csv")
IMG_OUTPUT = os.path.join(OUTPUT_DIR, "confusion_matrix.png")

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv(DATASET_PATH)

# Assumption: first 4 columns = features, last column = label
X = df.iloc[:, :4]
y = df.iloc[:, -1]

# =========================
# TRAIN / TEST SPLIT (70/30)
# =========================
stratify_arg = y if y.nunique() > 1 else None
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=stratify_arg
)

# =========================
# TRAIN MODEL
# =========================
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# =========================
# CONFUSION MATRIX
# =========================
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

# -------------------------
# 1) SAVE CONFUSION MATRIX AS CSV
# -------------------------
cm_df = pd.DataFrame(
    cm,
    index=[f"Actual_{c}" for c in model.classes_],
    columns=[f"Predicted_{c}" for c in model.classes_]
)

cm_df.to_csv(CSV_OUTPUT)
print(f"Confusion matrix saved as CSV to: {CSV_OUTPUT}")

# -------------------------
# 2) SAVE CONFUSION MATRIX AS IMAGE (PNG)
# -------------------------
fig, ax = plt.subplots()
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)
disp.plot(ax=ax, cmap="Blues", colorbar=True)

ax.set_title("Confusion Matrix Heatmap (Logistic Regression)")
ax.set_xlabel("Predicted Class")
ax.set_ylabel("Actual Class")

plt.tight_layout()
plt.savefig(IMG_OUTPUT, dpi=300)
plt.close()

print(f"Confusion matrix image saved to: {IMG_OUTPUT}")
