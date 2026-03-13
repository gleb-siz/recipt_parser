# Receipt Total Parser (XGBoost)

This repo trains a lightweight XGBoost classifier to identify the **total amount token** in receipt OCR output. The model treats each OCR token as a sample and predicts whether it represents the receipt total (binary classification).

**Key ideas**
- Convert OCR bounding boxes into spatial features (row/column position, ranks, proximity to “total” keywords).
- Add simple semantic features (token length, numeric pattern/value).
- Train an XGBoost classifier with class weighting and early stopping.

## Dataset
This project is built around the **SROIE 2019** receipt dataset (Scanned Receipts OCR and Information Extraction), commonly distributed on Kaggle. The expected folder layout matches the SROIE format:

- `data/SROIE2019/train/`
- `data/SROIE2019/test/`

Each split should contain:
- `img/` receipt images
- `box/` OCR tokens + bounding boxes
- `entities/` JSON labels (e.g., `total`)

A local `data/validation/` set can be used for notebook evaluation, following the same structure.

## ML Approach
The training pipeline (`src/pipeline.py`) performs:
1. OCR + label loading (`src/utils.py`)
2. Feature engineering (geometry, row/column, keyword proximity, numeric parsing)
3. Train/validation split with stratification
4. XGBoost training with class weighting
5. Evaluation + MLflow logging + saved model artifacts

Default training features live in:
- `src/pipeline.py` (`DEFAULT_CONFIG["features"]`)
- `src/utils.py` (feature generation logic)

## Usage Tips

### Install dependencies
```bash
pip install -r requirements.txt
```

### Train the model
```bash
python3 src/train.py --config configs/xgb_pipeline.json
```

Outputs:
- MLflow runs in `mlruns/`
- Best model saved to `models/xgb_parser/best.ubj` and archived per run

### Evaluate on local validation data
Open:
- `notebooks/xgb_validation_eval.ipynb`

This notebook loads the latest archived model, evaluates it on `data/validation`, renders a probe prediction on an image, and generates SHAP explanations.

### Notebook for end-to-end exploration
Open:
- `notebooks/recipt_classifier_xg_boost.ipynb`

This notebook shows the full feature engineering + training workflow and can be used as a sandbox for feature ideas.

## Repo Layout
- `src/pipeline.py`: training + evaluation pipeline with MLflow logging
- `src/utils.py`: data loading + feature engineering
- `configs/xgb_pipeline.json`: default training config
- `models/`: saved model artifacts
- `notebooks/`: exploration and evaluation notebooks

## Notes
- This is a token-level classifier; post-processing can pick the top-scoring token per receipt.
- Performance depends heavily on OCR quality and the presence of “total”-like keywords.
