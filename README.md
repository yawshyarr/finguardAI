# FinGuard AI

FinGuard AI is a fraud detection dashboard that combines a FastAPI prediction service, an XGBoost model, a Streamlit glassmorphism interface, Plotly analytics, and a compact public-data reference sample.

## Highlights

- Professional Streamlit frontend with glassmorphism styling and no emoji UI labels
- FastAPI backend with `/predict`, `/predict-batch`, `/metadata`, and `/health`
- XGBoost model scoring with risk bands and reason codes
- Batch CSV analyzer with downloadable scored output
- Portfolio analytics for fraud rate, exposure, merchant risk, and review queues
- Demo transaction ledger plus a public sample from the Hugging Face `alenc123/credit-card-fraud` dataset

## Project Structure

```text
FinGuardAI/
  backend/main.py
  frontend/app.py
  data/transactions.csv
  data/public_fraud_sample.csv
  data/processed.csv
  models/xgboost_fraud.pkl
  notebooks/analysis.ipynb
  requirements.txt
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The current workspace already has a virtual environment at `venv/`.

## Run The Backend

From the project root:

```bash
uvicorn backend.main:app --reload --port 8000
```

Backend docs:

```text
http://127.0.0.1:8000/docs
```

## Run The Frontend

Open a second terminal from the project root:

```bash
streamlit run frontend/app.py
```

The frontend defaults to:

```text
http://localhost:8501
```

## CSV Batch Format

The batch analyzer accepts CSV files with these columns:

```csv
transaction_id,amount,merchant,type
101,72000,Unknown,Transfer
102,799,Netflix,Subscription
```

## Dataset Notes

`data/transactions.csv` is a compact resume/demo ledger for predictable charts and live scoring.

`data/public_fraud_sample.csv` is a small sample fetched from the public Hugging Face dataset page:

```text
https://huggingface.co/datasets/alenc123/credit-card-fraud
```

The public dataset is useful for showing realistic transaction schema and external data awareness without committing the full dataset into the repository.
