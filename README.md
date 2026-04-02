# AMI Anemia Risk Site

Public-facing AMI anemia 1-year mortality risk calculator and reporting bundle.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/xiaoxiong0913/ami-anemia-risk-site)

## Runtime

- Framework: Flask
- Public APIs:
  - `GET /api/config`
  - `GET /healthz`
  - `POST /api/predict`
- Locked runtime model: `Champion model: XGBoost`

## Local Run

```bash
pip install -r requirements.txt
python app.py
```

## Render

This repository is prepared for a Render Python Web Service using the root-level `render.yaml`, `Procfile`, and `requirements.txt`.
