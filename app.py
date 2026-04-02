from __future__ import annotations

import json
import os
from pathlib import Path
from pathlib import PureWindowsPath
from typing import Any

import joblib
import pandas as pd
from flask import Flask, Response, jsonify, request


APP_DIR = Path(__file__).resolve().parent
CARD_PATH = APP_DIR / "clinical_web_model_card.json"
HTML_PATH = APP_DIR / "clinical_risk_comparison.html"


def _bundle_path(relative_path: str) -> Path:
    return APP_DIR.joinpath(*PureWindowsPath(relative_path).parts)


def _dense(matrix):
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    if hasattr(matrix, "todense"):
        return matrix.todense()
    return matrix


def _coerce_runtime_value(value: Any, feature_spec: dict[str, Any]) -> Any:
    if value in (None, "", "null"):
        return feature_spec["default"]
    if feature_spec["type"] == "select":
        option_values = [option["value"] for option in feature_spec.get("options", [])]
        if any(item in [0, 1, "0", "1"] for item in option_values):
            return float(value)
        return str(value)
    return float(value)


with CARD_PATH.open("r", encoding="utf-8") as fh:
    PAYLOAD = json.load(fh)
RAW_FEATURE_ORDER = list(PAYLOAD["prediction_model"]["raw_feature_order"])
SELECTED_TRANSFORMED_FEATURES = list(PAYLOAD["prediction_model"]["selected_transformed_features"])
FEATURE_SPECS = {item["key"]: item for item in PAYLOAD["prediction_model"]["input_features"]}
MODEL = None
PREPROCESSOR = None
TRANSFORMED_INDEX_MAP: dict[str, int] | None = None

app = Flask(__name__)


def _ensure_runtime_loaded() -> None:
    global MODEL, PREPROCESSOR, TRANSFORMED_INDEX_MAP
    if MODEL is not None and PREPROCESSOR is not None and TRANSFORMED_INDEX_MAP is not None:
        return
    PREPROCESSOR = joblib.load(_bundle_path(PAYLOAD["runtime"]["preprocessor_file"]))
    MODEL = joblib.load(_bundle_path(PAYLOAD["runtime"]["model_file"]))
    transformed_names = PREPROCESSOR.get_feature_names_out().tolist()
    TRANSFORMED_INDEX_MAP = {name: index for index, name in enumerate(transformed_names)}


def _prediction_frame(inputs: dict[str, Any]) -> pd.DataFrame:
    row = {feature: _coerce_runtime_value(inputs.get(feature), FEATURE_SPECS[feature]) for feature in RAW_FEATURE_ORDER}
    return pd.DataFrame([row], columns=RAW_FEATURE_ORDER)


def _predict(inputs: dict[str, Any]) -> dict[str, Any]:
    _ensure_runtime_loaded()
    frame = _prediction_frame(inputs)
    dense = _dense(PREPROCESSOR.transform(frame))
    selected_idx = [TRANSFORMED_INDEX_MAP[name] for name in SELECTED_TRANSFORMED_FEATURES]
    risk = float(MODEL.predict_proba(dense[:, selected_idx])[:, 1][0])

    top_drivers = ", ".join(item["label"] for item in PAYLOAD["prediction_model"]["top_drivers"][:3])
    model_name = PAYLOAD["prediction_model"]["name"]
    return {
        "prediction_model": {
            "title": f"{model_name} deployment model",
            "risk": risk,
            "details": [
                f"Model alignment: {PAYLOAD['prediction_model']['deployment_note']}",
                f"Input scope: {PAYLOAD['metadata']['prediction_model_feature_count']} harmonized first-day ICU variables",
                f"External validation AUROC: {PAYLOAD['prediction_model']['performance']['external_auroc']:.3f}",
                f"Leading drivers: {top_drivers}",
            ],
        }
    }


@app.get("/")
def index():
    return Response(HTML_PATH.read_text(encoding="utf-8"), mimetype="text/html")


@app.get("/api/config")
def config():
    payload = dict(PAYLOAD)
    payload.pop("runtime", None)
    return jsonify(payload)


@app.post("/api/predict")
def predict():
    request_payload = request.get_json(silent=True) or {}
    try:
        return jsonify(_predict(request_payload.get("inputs", {})))
    except Exception as exc:
        app.logger.exception("Prediction request failed.")
        return jsonify({"error": "prediction_failed", "detail": str(exc)}), 500


@app.get("/healthz")
def healthz():
    return jsonify({"status": "ok", "model": PAYLOAD["prediction_model"]["name"], "runtime_loaded": MODEL is not None})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8765")))
