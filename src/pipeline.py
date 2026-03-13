import argparse
import json
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from utils import get_data

DEFAULT_CONFIG = {
    "data": {
        "train_path": "data/SROIE2019/train",
        "test_path": "data/SROIE2019/test",
    },
    "features": [
        "file_aspect_ratio",
        "x_max",
        "token_width",
        "token_heigh",
        "aspect_ratio",
        "row",
        "col",
        "row_rank",
        "col_rank",
        "has_total_keyword_in_row",
        "tokens_in_col",
        "tokens_in_row",
        "text_length",
        "is_digit",
        "font_size",
        "row_dist_from_total",
        "value",
        "rows_in_col",
        "cols_in_row",
        "has_total_below",
    ],
    "train": {
        "val_size": 0.2,
        "random_state": 42,
        "verbose": 1,
    },
    "model": {
        "name": "xgb_parser",
        "models_path": "models",
        "xgb_params": {
            "objective": "binary:logistic",
            "n_estimators": 500,
            "learning_rate": 0.1,
            "max_depth": 5,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "use_label_encoder": False,
            "early_stopping_rounds": 50,
        },
    },
    "mlflow": {
        "tracking_uri": "file:./mlruns",
        "experiment_name": "receipt-xgb",
        "run_name": "xgb-baseline",
    },
}


def _deep_merge(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _repo_root():
    return Path(__file__).resolve().parent.parent


def _resolve_path(path_value):
    if os.path.isabs(path_value):
        return path_value
    return str(_repo_root() / path_value)


def load_config(config_path=None):
    cfg = deepcopy(DEFAULT_CONFIG)
    if config_path:
        with open(config_path, "r") as f:
            user_cfg = json.load(f)
        cfg = _deep_merge(cfg, user_cfg)

    cfg["data"]["train_path"] = _resolve_path(cfg["data"]["train_path"])
    cfg["data"]["test_path"] = _resolve_path(cfg["data"]["test_path"])
    cfg["model"]["models_path"] = _resolve_path(cfg["model"]["models_path"])

    return cfg


def load_data(cfg):
    X_train, y_train = get_data(cfg["data"]["train_path"], features=cfg["features"])
    X_test, y_test = get_data(cfg["data"]["test_path"], features=cfg["features"])
    return X_train, y_train, X_test, y_test


def split_train_val(X_train, y_train, cfg):
    return train_test_split(
        X_train,
        y_train,
        test_size=cfg["train"]["val_size"],
        stratify=y_train,
        random_state=cfg["train"]["random_state"],
    )


def train_model(X_train, y_train, X_val, y_val, cfg):
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, weights))
    sample_weights = np.array([class_weight_dict[label] for label in y_train])

    model = xgb.XGBClassifier(**cfg["model"]["xgb_params"])
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        sample_weight=sample_weights,
        verbose=cfg["train"]["verbose"],
    )
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_pred)
    report["ROC_AUC"] = roc_auc
    return report


def _predict_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X)


def _save_confusion_matrix(y_true, y_pred, title, out_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _save_roc_curve(y_true, y_score, title, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="#1f77b4", linewidth=2)
    ax.plot([0, 1], [0, 1], color="#666666", linestyle="--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _save_pr_curve(y_true, y_score, title, out_path):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, color="#ff7f0e", linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _save_feature_importance(model, feature_names, title, out_path, top_n=20):
    if not hasattr(model, "feature_importances_"):
        return False
    importances = model.feature_importances_
    if importances is None or len(importances) == 0:
        return False

    names = feature_names
    if names is None or len(names) != len(importances):
        names = [f"f{i}" for i in range(len(importances))]

    order = np.argsort(importances)[::-1][:top_n]
    ordered_importances = importances[order][::-1]
    ordered_names = [names[i] for i in order][::-1]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.barh(ordered_names, ordered_importances, color="#2ca02c")
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return True


def _flatten_metrics(report):
    metrics = {}
    for key, value in report.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                metrics[f"{key}.{sub_key}"] = float(sub_value)
        else:
            metrics[key] = float(value)
    return metrics


def save_model(model, cfg, run_id):
    models_path = Path(cfg["model"]["models_path"]) / cfg["model"]["name"]
    archive_path = models_path / "archive" / run_id
    archive_path.mkdir(parents=True, exist_ok=True)
    models_path.mkdir(parents=True, exist_ok=True)

    best_path = models_path / "best.ubj"
    archive_model_path = archive_path / "best.ubj"

    model.save_model(str(best_path))
    model.save_model(str(archive_model_path))

    return str(best_path), str(archive_model_path)


def run_pipeline(config_path=None):
    cfg = load_config(config_path)

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    X_train, y_train, X_test, y_test = load_data(cfg)
    X_tr, X_val, y_tr, y_val = split_train_val(X_train, y_train, cfg)

    run_id = datetime.now().strftime("%Y%m%d%H%M%S")

    with mlflow.start_run(run_name=cfg["mlflow"].get("run_name")):
        model = train_model(X_tr, y_tr, X_val, y_val, cfg)
        report = evaluate_model(model, X_test, y_test)
        metrics = _flatten_metrics(report)

        mlflow.log_params(cfg["model"]["xgb_params"])
        mlflow.log_params(
            {
                "features_count": len(cfg["features"]),
                "val_size": cfg["train"]["val_size"],
                "random_state": cfg["train"]["random_state"],
            }
        )
        mlflow.log_metrics(metrics)

        plots_dir = Path("/tmp") / "mlflow_artifacts" / run_id
        plots_dir.mkdir(parents=True, exist_ok=True)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        _save_confusion_matrix(
            y_train,
            y_pred_train,
            "Confusion Matrix (Train)",
            plots_dir / "confusion_matrix_train.png",
        )
        _save_confusion_matrix(
            y_test,
            y_pred_test,
            "Confusion Matrix (Test)",
            plots_dir / "confusion_matrix_test.png",
        )

        y_score_test = _predict_scores(model, X_test)
        _save_roc_curve(
            y_test,
            y_score_test,
            "ROC Curve (Test)",
            plots_dir / "roc_curve_test.png",
        )
        _save_pr_curve(
            y_test,
            y_score_test,
            "Precision-Recall Curve (Test)",
            plots_dir / "pr_curve_test.png",
        )
        _save_feature_importance(
            model,
            cfg["features"],
            "Feature Importance (Top 20)",
            plots_dir / "feature_importance.png",
        )

        mlflow.log_artifacts(str(plots_dir), artifact_path="plots")

        best_path, archive_path = save_model(model, cfg, run_id)
        mlflow.log_artifact(best_path, artifact_path="model")

        if config_path:
            mlflow.log_artifact(config_path, artifact_path="config")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate XGBoost receipt parser."
    )
    parser.add_argument(
        "--config", default="configs/xgb_pipeline.json", help="Path to JSON config."
    )
    args = parser.parse_args()

    report = run_pipeline(args.config)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
