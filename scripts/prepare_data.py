from pathlib import Path
import pandas as pd
from datasets import load_dataset

raw_dir = Path("data/raw")
processed_dir = Path("data/processed")

raw_dir.mkdir(parents=True, exist_ok=True)
processed_dir.mkdir(parents=True, exist_ok=True)

dataset = load_dataset("PolyAI/banking77")

label_names = dataset["train"].features["label"].names

train_df = pd.DataFrame(dataset["train"])
test_df = pd.DataFrame(dataset["test"])

train_df["split"] = "train"
test_df["split"] = "test"

df = pd.concat([train_df, test_df], ignore_index=True)
df = df.rename(columns={"text": "query"})
df["intent"] = df["label"].map(dict(enumerate(label_names)))

label_mapping = pd.DataFrame({
    "label": range(len(label_names)),
    "intent": label_names
})

df[["query", "label", "intent", "split"]].to_csv(
    processed_dir / "banking77.csv",
    index=False,
    encoding="utf-8"
)

label_mapping.to_csv(
    processed_dir / "label_mapping.csv",
    index=False,
    encoding="utf-8"
)

train_df.to_csv(raw_dir / "banking77_train.csv", index=False, encoding="utf-8")
test_df.to_csv(raw_dir / "banking77_test.csv", index=False, encoding="utf-8")
