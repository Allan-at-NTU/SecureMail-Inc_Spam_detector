# Spam Detector — Ready-to-Deploy Bundle

This repo contains:
- `notebooks/01_strategy_model_selection.ipynb`: tries multiple encoders/models, selects best, **saves to `models/`**.
- `notebooks/02_deploy_backend_and_ui.ipynb`: wires the trained model into a **FastAPI** app with a simple UI.
- `app/`: FastAPI backend, minimal UI (HTML/CSS), loads `models/best_model.joblib`.
- `scripts/train.py`: optional CLI trainer if you want a one-shot pipeline.
- `requirements.txt`, `Dockerfile`, `render.yaml`: ready for Render or any Docker-compatible host.

## Quick Start (Local)
1. (Optional) Create & activate a virtual env.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train & select best model using the strategy notebook **(run on your machine)**, which will create `models/best_model.joblib`.
   - Alternatively:
     ```bash
     python scripts/train.py --input_csv path/to/your.csv
     ```
4. Run the app:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```
5. Open http://localhost:8000

## Render Deploy
- Push this folder to GitHub.
- Create a new **Web Service** on Render, select your repo.
- Render will build from the `Dockerfile` and start Uvicorn on port 10000 (as configured).

## Notes
- Keep the environment **CPU-only**. This project avoids heavy GPU dependencies by default.
- For stronger models on large corpora, consider adding sentence-transformer encoders or fine-tuned transformers. You can extend the first notebook accordingly.

## Dataset
- Your dataset is included at `data/Spam_Detector.csv`.
- **Notebook 1** reads `Spam_Detector.csv` from the same folder as the notebook for convenience (a copy is placed in `notebooks/`).
- For production and version control, keep the canonical copy under `data/Spam_Detector.csv`.
