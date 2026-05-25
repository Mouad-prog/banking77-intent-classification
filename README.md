# Banking77 Intent Classification

This repository contains a Text Mining project for automatic intent classification on banking customer queries using the BANKING77 dataset.

The objective is to compare classical text representation methods, such as Bag of Words and TF-IDF, with semantic sentence embeddings based on Sentence-BERT.

## Project overview

The project studies how short banking queries can be automatically classified into one of 77 predefined intents.

The complete pipeline is implemented through Python scripts. The dataset is not included in the repository because it is automatically downloaded from Hugging Face when running the data preparation script.

## Dataset

The project uses the BANKING77 dataset.

Main characteristics:

- 13,083 banking customer queries
- 10,003 training examples
- 3,080 test examples
- 77 intent classes

The dataset is loaded automatically by:

```bash
python scripts/prepare_data.py
