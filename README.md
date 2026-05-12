# AI-Based Recruitment Fraud Detection

This Python project trains job-posting fraud classifiers that automatically label postings as `real` or `fake`.

It uses the public Real/Fake Job Posting Prediction dataset. By default, the script downloads a Hugging Face mirror of `fake_job_postings.csv`; you can also pass a local CSV with `--data-path`.

## Models

- TF-IDF + Logistic Regression baseline
- Bidirectional LSTM with Keras/TensorFlow
- BERT classification with Transformers/TensorFlow. The script first tries TensorFlow-compatible pretrained weights, then falls back to a compact TensorFlow-native BERT model when a checkpoint only exposes PyTorch/safetensors weights in the local environment.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run End-to-End

For a faster smoke test:

```powershell
python recruitment_fraud_detector.py --max-rows 1200 --lstm-epochs 1 --bert-epochs 1
```

For full training:

```powershell
python recruitment_fraud_detector.py
```

## Run the App

After training, start the Gradio app:

```powershell
python app.py
```

By default, the app uses `artifacts/` when present and falls back to `artifacts_smoke/`. To point it at another trained model directory:

```powershell
$env:ARTIFACTS_DIR="C:\path\to\artifacts"
python app.py
```

On Windows, TensorFlow may fail to install inside very deep directories unless long paths are enabled. A short virtual environment path, such as `C:\Users\prava\Documents\Codex\rf-fraud-venv`, avoids that issue.

If you already downloaded the Kaggle CSV:

```powershell
python recruitment_fraud_detector.py --data-path C:\path\to\fake_job_postings.csv
```

## Prediction

Training saves artifacts under `artifacts/`. You can run a prediction after training:

```powershell
python recruitment_fraud_detector.py --max-rows 1200 --lstm-epochs 1 --bert-epochs 1 --predict "Remote data entry role. No interview required. Pay a processing fee to receive the offer letter."
```

Or import the function:

```python
from recruitment_fraud_detector import predict_job_posting

result = predict_job_posting(
    "We are hiring a backend engineer with Python, SQL, and cloud experience.",
    model_type="lstm",
    artifacts_dir="artifacts",
)
print(result)
```

## Outputs

The script prints accuracy, F1 scores, confusion matrices, and a model comparison table. It also writes:

- `artifacts/metrics.json`
- `artifacts/lstm_model.keras`
- `artifacts/lstm_tokenizer.pkl`
- `artifacts/bert_model/`
- `artifacts/tfidf_model.pkl`
- `artifacts/tfidf_vectorizer.pkl`
