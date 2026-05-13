"""Gradio app for recruitment fraud detection."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

import gradio as gr
import pandas as pd

from recruitment_fraud_detector import predict_job_posting


ROOT = Path(__file__).resolve().parent
DEFAULT_ARTIFACTS_DIR = ROOT / "artifacts"
SMOKE_ARTIFACTS_DIR = ROOT / "artifacts_smoke"


def resolve_artifacts_dir() -> Path:
    configured = os.environ.get("ARTIFACTS_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    if DEFAULT_ARTIFACTS_DIR.exists():
        return DEFAULT_ARTIFACTS_DIR
    return SMOKE_ARTIFACTS_DIR


ARTIFACTS_DIR = resolve_artifacts_dir()


def available_models() -> List[str]:
    choices = []
    if (ARTIFACTS_DIR / "lstm_model.keras").exists():
        choices.append("lstm")
    if (ARTIFACTS_DIR / "bert_model").exists():
        choices.append("bert")
    return choices or ["lstm"]


def load_metrics_table() -> pd.DataFrame:
    metrics_path = ARTIFACTS_DIR / "metrics.json"
    if not metrics_path.exists():
        return pd.DataFrame(columns=["model", "accuracy", "fake_f1", "macro_f1"])

    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    return pd.DataFrame(
        [
            {
                "model": item["model"],
                "accuracy": round(float(item["accuracy"]), 4),
                "fake_f1": round(float(item["fake_f1"]), 4),
                "macro_f1": round(float(item["macro_f1"]), 4),
            }
            for item in metrics
        ]
    )


def classify_job(description: str, model_type: str) -> Dict[str, float]:
    if not description or not description.strip():
        raise gr.Error("Paste a job description first.")
    try:
        result = predict_job_posting(
            description,
            model_type=model_type,
            artifacts_dir=str(ARTIFACTS_DIR),
        )
    except FileNotFoundError as exc:
        raise gr.Error(
            f"Missing model artifacts in {ARTIFACTS_DIR}. Run training first."
        ) from exc

    return {
        "fake": result["fake_probability"],
        "real": result["real_probability"],
    }


EXAMPLES = [
    [
        "Remote data entry role. No interview required. Pay a processing fee to receive the offer letter.",
        "lstm",
    ],
    [
        "We are hiring a backend engineer with Python, SQL, cloud experience, technical interviews, and full benefits.",
        "lstm",
    ],
]


CSS = """
body, .gradio-container {
    background: #f7f7f2;
}
.gradio-container {
    max-width: 1120px !important;
}
#title {
    margin-bottom: 8px;
}
button.primary {
    border-radius: 8px !important;
}
"""


with gr.Blocks(title="Recruitment Fraud Detector") as demo:
    gr.Markdown("# Recruitment Fraud Detector", elem_id="title")

    with gr.Row(equal_height=False):
        with gr.Column(scale=2):
            job_text = gr.Textbox(
                label="Job description",
                lines=14,
                placeholder="Paste the full job posting here...",
            )
            with gr.Row():
                model_choice = gr.Radio(
                    choices=available_models(),
                    value=available_models()[0],
                    label="Model",
                )
                classify_button = gr.Button("Classify", variant="primary")

        with gr.Column(scale=1):
            prediction = gr.Label(label="Prediction", num_top_classes=2)
            metrics = gr.Dataframe(
                value=load_metrics_table(),
                label="Model comparison",
                interactive=False,
                wrap=True,
            )

    gr.Examples(
        examples=EXAMPLES,
        inputs=[job_text, model_choice],
        label="Examples",
    )

    classify_button.click(
        fn=classify_job,
        inputs=[job_text, model_choice],
        outputs=prediction,
        api_name="classify_job",
    )
    job_text.submit(
        fn=classify_job,
        inputs=[job_text, model_choice],
        outputs=prediction,
        api_name=False,
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, css=CSS)
