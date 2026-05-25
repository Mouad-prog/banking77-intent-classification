from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix

data_path = Path("data/processed/banking77.csv")
model_dir = Path("models")
fig_dir = Path("figures")
res_dir = Path("results")

fig_dir.mkdir(parents=True, exist_ok=True)
res_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(data_path)

train_df = df[df["split"] == "train"].copy()
test_df = df[df["split"] == "test"].copy()

X_train = np.load(model_dir / "sbert_train_embeddings.npy")
X_test = np.load(model_dir / "sbert_test_embeddings.npy")

y_train = train_df["label"].values
y_test = test_df["label"].values

label_table = (
    df[["label", "intent"]]
    .drop_duplicates()
    .sort_values("label")
)

label_to_intent = dict(zip(label_table["label"], label_table["intent"]))
label_names = label_table["intent"].tolist()

model = LinearSVC(class_weight="balanced")
model.fit(X_train, y_train)
pred = model.predict(X_test)

report = classification_report(
    y_test,
    pred,
    target_names=label_names,
    output_dict=True,
    zero_division=0
)

per_class = (
    pd.DataFrame(report)
    .T
    .reset_index()
    .rename(columns={"index": "intent"})
)

per_class = per_class[
    ~per_class["intent"].isin(["accuracy", "macro avg", "weighted avg"])
]

per_class = per_class.sort_values("f1-score")
per_class.to_csv(res_dir / "sbert_svm_per_class_report.csv", index=False)

cm = confusion_matrix(y_test, pred)
pairs = []

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        if i != j and cm[i, j] > 0:
            pairs.append({
                "true_label": i,
                "predicted_label": j,
                "true_intent": label_to_intent[i],
                "predicted_intent": label_to_intent[j],
                "count": int(cm[i, j])
            })

confusions = pd.DataFrame(pairs).sort_values("count", ascending=False)
confusions.to_csv(res_dir / "top_confused_intents.csv", index=False)

test_errors = test_df.copy()
test_errors["predicted_label"] = pred
test_errors["predicted_intent"] = test_errors["predicted_label"].map(label_to_intent)
test_errors = test_errors[test_errors["label"] != test_errors["predicted_label"]]
test_errors[["query", "intent", "predicted_intent"]].to_csv(
    res_dir / "sbert_svm_errors.csv",
    index=False
)

worst = per_class.head(15)

plt.figure(figsize=(9, 6))
plt.barh(worst["intent"][::-1], worst["f1-score"][::-1])
plt.title("Intentions les plus difficiles - SBERT + SVM")
plt.xlabel("F1-score")
plt.tight_layout()
plt.savefig(fig_dir / "08_sbert_svm_worst_intents.png", dpi=300)
plt.close()

top_confusions = confusions.head(15)

plt.figure(figsize=(10, 6))
labels = [
    f"{row.true_intent}\n→ {row.predicted_intent}"
    for _, row in top_confusions.iterrows()
]
plt.barh(labels[::-1], top_confusions["count"][::-1])
plt.title("Principales confusions entre intentions")
plt.xlabel("Nombre d'erreurs")
plt.tight_layout()
plt.savefig(fig_dir / "09_top_confused_intents.png", dpi=300)
plt.close()
