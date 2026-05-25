from pathlib import Path
import re
import pandas as pd
import nltk
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, ConfusionMatrixDisplay

nltk.download("stopwords", quiet=True)

data_path = Path("data/processed/banking77.csv")
processed_path = Path("data/processed/banking77_clean.csv")
fig_dir = Path("figures")
res_dir = Path("results")

fig_dir.mkdir(parents=True, exist_ok=True)
res_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(data_path)

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    tokens = [stemmer.stem(t) for t in tokens]
    return " ".join(tokens)

df["clean_query"] = df["query"].apply(clean_text)
df.to_csv(processed_path, index=False, encoding="utf-8")

train_df = df[df["split"] == "train"].copy()
test_df = df[df["split"] == "test"].copy()

X_train = train_df["clean_query"]
y_train = train_df["label"]
X_test = test_df["clean_query"]
y_test = test_df["label"]

label_names = (
    df[["label", "intent"]]
    .drop_duplicates()
    .sort_values("label")["intent"]
    .tolist()
)

models = {
    "bow_logreg": Pipeline([
        ("vectorizer", CountVectorizer(min_df=2, ngram_range=(1, 2))),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ]),
    "bow_svm": Pipeline([
        ("vectorizer", CountVectorizer(min_df=2, ngram_range=(1, 2))),
        ("classifier", LinearSVC(class_weight="balanced"))
    ]),
    "tfidf_logreg": Pipeline([
        ("vectorizer", TfidfVectorizer(min_df=2, ngram_range=(1, 2))),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ]),
    "tfidf_svm": Pipeline([
        ("vectorizer", TfidfVectorizer(min_df=2, ngram_range=(1, 2))),
        ("classifier", LinearSVC(class_weight="balanced"))
    ]),
}

rows = []

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        pred,
        average="macro",
        zero_division=0
    )

    rows.append({
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1
    })

    report = classification_report(
        y_test,
        pred,
        target_names=label_names,
        zero_division=0
    )

    with open(res_dir / f"classification_report_{name}.txt", "w", encoding="utf-8") as f:
        f.write(report)

    fig, ax = plt.subplots(figsize=(12, 10))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        pred,
        ax=ax,
        cmap=None,
        colorbar=False,
        xticks_rotation="vertical"
    )
    ax.set_title(f"Matrice de confusion - {name}")
    plt.tight_layout()
    plt.savefig(fig_dir / f"confusion_matrix_{name}.png", dpi=300)
    plt.close()

results = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
results.to_csv(res_dir / "baseline_model_comparison.csv", index=False)

plt.figure(figsize=(8, 4))
plt.bar(results["model"], results["macro_f1"])
plt.title("Comparaison des modèles baseline - Macro F1")
plt.xlabel("Modèle")
plt.ylabel("Macro F1-score")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(fig_dir / "04_baseline_macro_f1_comparison.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 4))
plt.bar(results["model"], results["accuracy"])
plt.title("Comparaison des modèles baseline - Accuracy")
plt.xlabel("Modèle")
plt.ylabel("Accuracy")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(fig_dir / "05_baseline_accuracy_comparison.png", dpi=300)
plt.close()
