from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_CSV = "psd_specparam_features_0_90Hz.csv"
PREDICTIONS_CSV = "madrs_loo_predictions_0_90Hz.csv"
METRICS_CSV = "madrs_loo_metrics_0_90Hz.csv"
TIMELINE_PDF = "madrs_predictions_over_time_0_90Hz.pdf"
REGRESSION_PDF = "madrs_prediction_regression_0_90Hz.pdf"


def run_loo(df_hemi: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    X = df_hemi[feature_cols].to_numpy()
    y = df_hemi["madrs"].to_numpy(dtype=float)

    loo = LeaveOneOut()
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))

    y_pred = np.zeros_like(y, dtype=float)
    for train_idx, test_idx in loo.split(X):
        model.fit(X[train_idx], y[train_idx])
        y_pred[test_idx[0]] = model.predict(X[test_idx])[0]

    df_pred = df_hemi[["date", "hemisphere", "channel_name", "madrs"]].copy()
    df_pred = df_pred.rename(columns={"madrs": "madrs_true"})
    df_pred["madrs_pred"] = y_pred
    return df_pred


def safe_pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    return float(r2_score(y_true, y_pred))


def main() -> None:
    feature_path = Path(FEATURE_CSV)
    if not feature_path.exists():
        raise FileNotFoundError(
            f"Could not find {FEATURE_CSV}. Run run_neural_cv.py first."
        )

    df = pd.read_csv(feature_path)
    df["date"] = pd.to_datetime(df["date"], format="%Y_%m_%d")
    df = df.sort_values(["date", "hemisphere"]).reset_index(drop=True)

    # Include all requested spectral features and aperiodic parameters.
    feature_cols = [
        "aperiodic_offset",
        "aperiodic_exponent",
        "raw_theta",
        "raw_alpha",
        "raw_low_beta",
        "raw_high_beta",
        "ap_theta",
        "ap_alpha",
        "ap_low_beta",
        "ap_high_beta",
    ]

    missing_features = [c for c in feature_cols + ["madrs"] if c not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required columns in feature CSV: {missing_features}")

    valid_mask = df[feature_cols + ["madrs", "hemisphere"]].notna().all(axis=1)
    df_model = df.loc[valid_mask].copy().reset_index(drop=True)

    pred_parts = []
    metric_rows = []
    hemispheres = ["left", "right"]

    for hemisphere in hemispheres:
        df_hemi = df_model[df_model["hemisphere"].str.lower() == hemisphere].copy()
        if df_hemi.empty:
            continue

        df_hemi_pred = run_loo(df_hemi, feature_cols)
        pred_parts.append(df_hemi_pred)

        y_true = df_hemi_pred["madrs_true"].to_numpy(dtype=float)
        y_pred = df_hemi_pred["madrs_pred"].to_numpy(dtype=float)

        metric_rows.append(
            {
                "hemisphere": hemisphere,
                "pearson_r": safe_pearson_r(y_true, y_pred),
                "r2": safe_r2(y_true, y_pred),
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "mse": float(mean_squared_error(y_true, y_pred)),
                "n_samples": len(df_hemi_pred),
            }
        )

    if not pred_parts:
        raise ValueError("No valid samples available to run hemisphere-specific LOO CV.")

    df_pred = pd.concat(pred_parts, ignore_index=True)
    df_pred = df_pred.sort_values(["hemisphere", "date"]).reset_index(drop=True)
    df_pred.to_csv(PREDICTIONS_CSV, index=False)

    df_metrics = pd.DataFrame(metric_rows)
    df_metrics.to_csv(METRICS_CSV, index=False)

    # Plot 1: true vs predicted over time, split by hemisphere.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for idx, hemisphere in enumerate(hemispheres):
        ax = axes[idx]
        hemi_pred = df_pred[df_pred["hemisphere"].str.lower() == hemisphere].copy()
        hemi_pred = hemi_pred.sort_values("date")

        x = np.arange(len(hemi_pred))
        ax.plot(x, hemi_pred["madrs_true"], marker="o", label="True MADRS", color="black")
        ax.plot(x, hemi_pred["madrs_pred"], marker="o", label="Pred MADRS", color="tab:blue")

        date_labels = [d.strftime("%Y-%m-%d") for d in hemi_pred["date"]]
        ax.set_xticks(x)
        ax.set_xticklabels(date_labels, rotation=45, ha="right")
        ax.set_xlabel("Sample (ordered by date)")
        if idx == 0:
            ax.set_ylabel("MADRS")
        ax.set_title(f"{hemisphere.capitalize()} Hemisphere")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")

    fig.suptitle("Leave-One-Sample-Out Predictions Over Time")
    fig.tight_layout()
    fig.savefig(TIMELINE_PDF)
    plt.close(fig)

    # Plot 2: regression plots split by hemisphere.
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    for idx, hemisphere in enumerate(hemispheres):
        ax = axes[idx]
        hemi_pred = df_pred[df_pred["hemisphere"].str.lower() == hemisphere]

        y_true = hemi_pred["madrs_true"].to_numpy(dtype=float)
        y_pred = hemi_pred["madrs_pred"].to_numpy(dtype=float)
        ax.scatter(y_true, y_pred, color="tab:blue", alpha=0.85)

        if len(y_true) > 0:
            y_min = min(y_true.min(), y_pred.min())
            y_max = max(y_true.max(), y_pred.max())
            ax.plot([y_min, y_max], [y_min, y_max], color="black", linestyle="--", label="Identity")

        if len(y_true) >= 2:
            fit_coef = np.polyfit(y_true, y_pred, deg=1)
            fit_line_x = np.linspace(y_true.min(), y_true.max(), 100)
            fit_line_y = fit_coef[0] * fit_line_x + fit_coef[1]
            ax.plot(fit_line_x, fit_line_y, color="tab:red", label="Best fit")

        ax.set_xlabel("True MADRS")
        if idx == 0:
            ax.set_ylabel("Predicted MADRS")
        ax.set_title(f"{hemisphere.capitalize()} Hemisphere")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")

    fig.suptitle("LOO Prediction Regression Plot")
    fig.tight_layout()
    fig.savefig(REGRESSION_PDF)
    plt.close(fig)

    print("Hemisphere-specific LOO CV complete")
    for _, row in df_metrics.iterrows():
        print(
            f"{row['hemisphere']}: "
            f"Pearson r={row['pearson_r']:.4f}, "
            f"R2={row['r2']:.4f}, "
            f"MAE={row['mae']:.4f}, "
            f"MSE={row['mse']:.4f}, "
            f"n={int(row['n_samples'])}"
        )
    print(f"Saved: {PREDICTIONS_CSV}")
    print(f"Saved: {METRICS_CSV}")
    print(f"Saved: {TIMELINE_PDF}")
    print(f"Saved: {REGRESSION_PDF}")


if __name__ == "__main__":
    main()
