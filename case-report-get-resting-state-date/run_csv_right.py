from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_CSV = "psd_specparam_features_0_40Hz.csv"
METRICS_CSV = "run_csv_right.csv"
PREDICTIONS_CSV = "run_csv_right_predictions.csv"
TIMELINE_PDF = "run_csv_right_lineplot.pdf"
REGRESSION_PDF = "run_csv_right_regplot.pdf"

FEATURE_COLS = ["raw_low_beta", "raw_high_beta", "aperiodic_offset"]


def main() -> None:
    feature_path = Path(FEATURE_CSV)
    if not feature_path.exists():
        raise FileNotFoundError(
            f"Could not find {FEATURE_CSV}. Run run_neural_cv.py first."
        )

    df = pd.read_csv(feature_path)
    df = df[df["hemisphere"].str.lower() == "right"].copy()
    df["date"] = pd.to_datetime(df["date"], format="%Y_%m_%d")
    df = df.sort_values("date").reset_index(drop=True)

    valid_mask = df[FEATURE_COLS + ["madrs"]].notna().all(axis=1)
    df = df.loc[valid_mask].reset_index(drop=True)

    X = df[FEATURE_COLS].to_numpy()
    y = df["madrs"].to_numpy(dtype=float)

    loo = LeaveOneOut()
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))

    y_pred = np.zeros_like(y, dtype=float)
    for train_idx, test_idx in loo.split(X):
        model.fit(X[train_idx], y[train_idx])
        y_pred[test_idx[0]] = model.predict(X[test_idx])[0]

    y_pred = np.clip(y_pred, 0, None)

    pearson_r = float(np.corrcoef(y, y_pred)[0, 1]) if len(y) > 1 else float("nan")
    r2 = float(r2_score(y, y_pred)) if len(y) > 1 else float("nan")
    mae = float(mean_absolute_error(y, y_pred))
    mse = float(mean_squared_error(y, y_pred))

    df_pred = df[["date", "hemisphere", "channel_name", "madrs"]].copy()
    df_pred = df_pred.rename(columns={"madrs": "madrs_true"})
    df_pred["madrs_pred"] = y_pred
    df_pred.to_csv(PREDICTIONS_CSV, index=False)

    df_metrics = pd.DataFrame(
        [
            {
                "hemisphere": "right",
                "features": ",".join(FEATURE_COLS),
                "prediction_postprocess": "clip_min_0",
                "pearson_r": pearson_r,
                "r2": r2,
                "mae": mae,
                "mse": mse,
                "n_samples": len(df),
            }
        ]
    )
    df_metrics.to_csv(METRICS_CSV, index=False)

    metrics_line = (
        f"Pearson r={pearson_r:.4f} | R2={r2:.4f} | "
        f"MAE={mae:.4f} | MSE={mse:.4f}"
    )

    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(df_pred))
    ax.plot(x, df_pred["madrs_true"], marker="o", color="black", label="True MADRS")
    ax.plot(x, df_pred["madrs_pred"], marker="o", color="tab:blue", label="Pred MADRS (clipped)")
    ax.set_xticks(x)
    ax.set_xticklabels([d.strftime("%Y-%m-%d") for d in df_pred["date"]], rotation=45, ha="right")
    ax.set_xlabel("Date")
    ax.set_ylabel("MADRS")
    ax.set_title("Right Hemisphere LOO Predictions Over Time\n" + metrics_line)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(TIMELINE_PDF)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df_pred["madrs_true"], df_pred["madrs_pred"], color="tab:blue", alpha=0.85)
    y_min = min(df_pred["madrs_true"].min(), df_pred["madrs_pred"].min())
    y_max = max(df_pred["madrs_true"].max(), df_pred["madrs_pred"].max())
    ax.plot([y_min, y_max], [y_min, y_max], "k--", label="Identity")

    if len(df_pred) >= 2:
        fit_coef = np.polyfit(df_pred["madrs_true"], df_pred["madrs_pred"], deg=1)
        fit_line_x = np.linspace(y_min, y_max, 100)
        fit_line_y = fit_coef[0] * fit_line_x + fit_coef[1]
        ax.plot(fit_line_x, fit_line_y, color="tab:red", label="Best fit")

    ax.set_xlabel("True MADRS")
    ax.set_ylabel("Predicted MADRS")
    ax.set_title("Right Hemisphere LOO Regression Plot\n" + metrics_line)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(REGRESSION_PDF)
    plt.close(fig)

    print("Right-hemisphere LOO CV complete")
    print(metrics_line)
    print(f"Saved: {METRICS_CSV}")
    print(f"Saved: {PREDICTIONS_CSV}")
    print(f"Saved: {TIMELINE_PDF}")
    print(f"Saved: {REGRESSION_PDF}")


if __name__ == "__main__":
    main()