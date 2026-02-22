from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import joblib


def train_and_save(dataset_path: Path, task: str, output_dir: Path) -> None:
    df = pd.read_csv(dataset_path)

    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    model.fit(df["text"], df["label"])

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{task}_detector.joblib"
    joblib.dump(model, out_file)
    print(f"Saved {task} model -> {out_file}")


def main() -> None:
    data_dir = Path("src/data")
    models_dir = Path("models")

    train_and_save(data_dir / "spam_dataset.csv", "spam", models_dir)
    train_and_save(data_dir / "fake_news_dataset.csv", "fake_news", models_dir)


if __name__ == "__main__":
    main()
