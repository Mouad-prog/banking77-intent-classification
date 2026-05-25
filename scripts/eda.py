from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

data_path = Path("data/processed/banking77.csv")
fig_dir = Path("figures")
res_dir = Path("results")

fig_dir.mkdir(parents=True, exist_ok=True)
res_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(data_path)

df["query_length"] = df["query"].str.len()
df["word_count"] = df["query"].str.split().str.len()

summary = pd.DataFrame([{
    "n_queries": len(df),
    "n_train": int((df["split"] == "train").sum()),
    "n_test": int((df["split"] == "test").sum()),
    "n_intents": df["intent"].nunique(),
    "avg_query_length": round(df["query_length"].mean(), 2),
    "avg_word_count": round(df["word_count"].mean(), 2),
    "min_word_count": int(df["word_count"].min()),
    "max_word_count": int(df["word_count"].max())
}])

summary.to_csv(res_dir / "eda_summary.csv", index=False)

intent_dist = (
    df["intent"]
    .value_counts()
    .rename_axis("intent")
    .reset_index(name="count")
)

intent_dist.to_csv(res_dir / "intent_distribution.csv", index=False)

plt.figure(figsize=(6, 4))
df["split"].value_counts().plot(kind="bar")
plt.title("Distribution train/test")
plt.xlabel("Split")
plt.ylabel("Nombre de requêtes")
plt.tight_layout()
plt.savefig(fig_dir / "01_split_distribution.png", dpi=300)
plt.close()

plt.figure(figsize=(7, 4))
df["word_count"].plot(kind="hist", bins=30)
plt.title("Distribution du nombre de mots par requête")
plt.xlabel("Nombre de mots")
plt.ylabel("Fréquence")
plt.tight_layout()
plt.savefig(fig_dir / "02_query_length_distribution.png", dpi=300)
plt.close()

top_20 = intent_dist.head(20)

plt.figure(figsize=(9, 6))
plt.barh(top_20["intent"][::-1], top_20["count"][::-1])
plt.title("Top 20 des intentions les plus fréquentes")
plt.xlabel("Nombre de requêtes")
plt.tight_layout()
plt.savefig(fig_dir / "03_top_20_intents.png", dpi=300)
plt.close()
