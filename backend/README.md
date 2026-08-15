# CardioVision Backend

FastAPI server that loads `models/checkpoints/best_model.pth`, runs the existing CardioViT inference pipeline, and returns frontend-ready JSON.

## Run

```bash
pip install -r requirements.txt
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

The frontend expects `NEXT_PUBLIC_API_URL=http://127.0.0.1:8000` by default.
