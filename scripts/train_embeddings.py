from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, ConfusionMatrixDisplay

data_path = Path("data/processed/banking77.csv")
fig_dir = Path("figures")
res_dir = Path("results")
model_dir = Path("models")

fig_dir.mkdir(parents=True, exist_ok=True)
res_dir.mkdir(parents=True, exist_ok=True)
model_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(data_path)

train_df = df[df["split"] == "train"].copy()
test_df = df[df["split"] == "test"].copy()

X_train_text = train_df["query"].tolist()
X_test_text = test_df["query"].tolist()
y_train = train_df["label"].values
y_test = test_df["label"].values

label_names = (
    df[["label", "intent"]]
    .drop_duplicates()
    .sort_values("label")["intent"]
    .tolist()
)

train_emb_path = model_dir / "sbert_train_embeddings.npy"
test_emb_path = model_dir / "sbert_test_embeddings.npy"

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

if train_emb_path.exists() and test_emb_path.exists():
    X_train = np.load(train_emb_path)
    X_test = np.load(test_emb_path)
else:
    X_train = encoder.encode(
        X_train_text,
        batch_size=128,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    X_test = encoder.encode(
        X_test_text,
        batch_size=128,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    np.save(train_emb_path, X_train)
    np.save(test_emb_path, X_test)

models = {
    "sbert_logreg": LogisticRegression(max_iter=3000, class_weight="balanced"),
    "sbert_svm": LinearSVC(class_weight="balanced")
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

embedding_results = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
embedding_results.to_csv(res_dir / "embedding_model_comparison.csv", index=False)

baseline = pd.read_csv(res_dir / "baseline_model_comparison.csv")
all_results = pd.concat([baseline, embedding_results], ignore_index=True)
all_results = all_results.sort_values("macro_f1", ascending=False)
all_results.to_csv(res_dir / "all_model_comparison.csv", index=False)

plt.figure(figsize=(10, 4))
plt.bar(all_results["model"], all_results["macro_f1"])
plt.title("Comparaison globale des modèles - Macro F1")
plt.xlabel("Modèle")
plt.ylabel("Macro F1-score")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(fig_dir / "06_all_models_macro_f1_comparison.png", dpi=300)
plt.close()

plt.figure(figsize=(10, 4))
plt.bar(all_results["model"], all_results["accuracy"])
plt.title("Comparaison globale des modèles - Accuracy")
plt.xlabel("Modèle")
plt.ylabel("Accuracy")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(fig_dir / "07_all_models_accuracy_comparison.png", dpi=300)
plt.close()
