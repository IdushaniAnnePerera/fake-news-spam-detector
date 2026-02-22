# Fake News / Spam Detection (NLP)

A **free, end-to-end Python project** for text classification with:

- ✅ Input text/news/email
- ✅ Predict **Real vs Fake** or **Spam vs Not Spam**
- ✅ Confidence score
- ✅ Classical NLP with **Scikit-learn**
- ✅ Optional **Transformers zero-shot** inference fallback
- ✅ Web app UI (FastAPI + HTML)

## 1) Project Structure

```text
.
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── models/
│   ├── spam_detector.joblib
│   └── fake_news_detector.joblib
├── src/
│   ├── detector.py
│   ├── transformer_detector.py
│   └── data/
│       ├── spam_dataset.csv
│       └── fake_news_dataset.csv
├── templates/
│   └── index.html
└── tests/
    └── test_detector.py
```

## 2) Quick Start

### Create virtual env and install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Train local Scikit-learn models

```bash
python train.py
```

### Run web app

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Open: `http://localhost:8000`

---

## 3) API Usage

### Health

```bash
curl http://localhost:8000/health
```

### Predict Spam

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"task":"spam","text":"Congratulations! You won a free iPhone. Click now.","use_transformer":false}'
```

### Predict Fake News

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"task":"fake_news","text":"Scientists confirm water found on Mars in latest mission.","use_transformer":false}'
```

---

## 4) Notes on Model Choices

- **Scikit-learn models** are trained locally on included starter datasets for free.
- **Transformer mode** uses Hugging Face `facebook/bart-large-mnli` as zero-shot classifier.
  - First run may download model weights.
  - If unavailable/offline, app gracefully falls back to sklearn.

---

## 5) Extend With Bigger Datasets

To improve quality:

- Replace `src/data/spam_dataset.csv` with SMS Spam Collection or Enron spam data.
- Replace `src/data/fake_news_dataset.csv` with larger fake/real news datasets.
- Re-run `python train.py`.

---

## 6) License

MIT (free to use and modify).
