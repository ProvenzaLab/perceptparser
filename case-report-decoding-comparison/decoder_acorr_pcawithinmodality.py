import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
import numpy as np
from sklearn import metrics
from sklearn.decomposition import PCA
# import svr
from sklearn.svm import SVR


audio_features = [
"F0semitoneFrom27.5Hz_sma3nz_amean",
"F0semitoneFrom27.5Hz_sma3nz_stddevNorm",
"F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2",
"F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope",
"F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope",
"F0semitoneFrom27.5Hz_sma3nz_stddevFallingSlope",
"F1frequency_sma3nz_amean",
"F1frequency_sma3nz_stddevNorm",
"F1bandwidth_sma3nz_amean",
"F1bandwidth_sma3nz_stddevNorm",
"F2frequency_sma3nz_amean",
"F2frequency_sma3nz_stddevNorm",
"F2bandwidth_sma3nz_amean",
"F2bandwidth_sma3nz_stddevNorm",
"F3frequency_sma3nz_amean",
"F3frequency_sma3nz_stddevNorm",
"F3bandwidth_sma3nz_amean",
"F3bandwidth_sma3nz_stddevNorm",
"VoicedSegmentsPerSec",
"MeanVoicedSegmentLengthSec",
"StddevVoicedSegmentLengthSec",
"MeanUnvoicedSegmentLength",
"StddevUnvoicedSegmentLength",
"valence",
"arousal",
"dominance",
]


df_video_allFAU = pd.read_csv("case-report-decoding-comparison/mean_fau_features_per_date.csv")
df_video_allFAU["date"] = pd.to_datetime(df_video_allFAU["date"], format="%Y-%m-%d")
df_video_allFAU = df_video_allFAU.drop(columns=["MADRS"])

df_audio = pd.read_csv("case-report-decoding-comparison/mean_audio_features_per_date.csv")
df_audio["date"] = pd.to_datetime(df_audio["date"], format="%Y-%m-%d")

df_video = pd.read_csv(
    "/scratch/timonmerk/get_data_NBU/perceptparser/case-report-decoding-comparison/video_loocv_cluster_prop_date_units.csv"
)
df_video = df_video.drop(columns=["n_csv_in_date", "n_frames_total", "env", "unit_id"])
df_video["date"] = pd.to_datetime(df_video["date"], format="%Y%m%d")
df_video = pd.merge(df_video, df_video_allFAU, on="date", how="inner")

df_neural = pd.read_csv("psd_specparam_features_0_40Hz.csv")
df_neural["date"] = pd.to_datetime(df_neural["date"], format="%Y_%m_%d")
df_neural = df_neural.rename(columns={"madrs": "MADRS"})
df_neural_r = df_neural[df_neural["hemisphere"].str.lower() == "right"].copy()
df_neural_r = df_neural_r.drop(columns=["hemisphere", "nbu_visit", "channel_name"])


def available_features(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [col for col in cols if col in df.columns]


def build_within_modality_pca_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    modality_feature_map: dict[str, list[str]],
    n_components: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    train_parts = []
    test_parts = []

    for modality_name, feature_cols in modality_feature_map.items():
        if not feature_cols:
            continue

        X_train_modality = train_df[feature_cols].to_numpy()
        X_test_modality = test_df[feature_cols].to_numpy()

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_modality)
        X_test_scaled = scaler.transform(X_test_modality)

        n_components_use = min(
            n_components,
            X_train_scaled.shape[1],
            X_train_scaled.shape[0],
        )

        if n_components_use < 1:
            continue

        pca = PCA(n_components=n_components_use)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)

        train_parts.append(X_train_pca)
        test_parts.append(X_test_pca)

    if not train_parts:
        raise ValueError("No features available to build within-modality PCA features.")

    return np.concatenate(train_parts, axis=1), np.concatenate(test_parts, axis=1)


def run_loso(
    df: pd.DataFrame,
    target_col: str,
    modality_feature_map: dict[str, list[str]] | None = None,
    use_within_modality_pca: bool = False,
    n_components: int = 2,
) -> tuple[pd.DataFrame, float, float, float, float]:
    df = df.sort_values("date").reset_index(drop=True)

    y = df[target_col].to_numpy(dtype=float)
    y_pred = []

    feature_cols_all = df.columns.difference(["date", target_col]).tolist()

    for i in range(len(df)):
        train_df = df.drop(index=i).reset_index(drop=True)
        test_df = df.iloc[[i]].copy()

        y_train = np.delete(y, i, axis=0)

        if use_within_modality_pca:
            if modality_feature_map is None:
                raise ValueError("modality_feature_map is required when use_within_modality_pca=True")

            X_train_use, X_test_use = build_within_modality_pca_features(
                train_df=train_df,
                test_df=test_df,
                modality_feature_map=modality_feature_map,
                n_components=n_components,
            )
        else:
            X_train = train_df[feature_cols_all].to_numpy()
            X_test = test_df[feature_cols_all].to_numpy()

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            X_train_use = X_train_scaled
            X_test_use = X_test_scaled

        model = linear_model.Ridge(alpha=1.0)
        model.fit(X_train_use, y_train)
        y_pred.append(model.predict(X_test_use)[0])

    df_pred = df.copy()
    df_pred["MADRS_pred"] = np.clip(y_pred, a_min=0, a_max=None)

    corr = df_pred[[target_col, "MADRS_pred"]].corr().iloc[0, 1]
    r2_ = metrics.r2_score(df_pred[target_col], df_pred["MADRS_pred"])
    mae = metrics.mean_absolute_error(df_pred[target_col], df_pred["MADRS_pred"])
    mse = metrics.mean_squared_error(df_pred[target_col], df_pred["MADRS_pred"])
    return df_pred, corr, r2_, mae, mse


audio_cols = available_features(df_audio, audio_features)
video_cols = [c for c in df_video.columns if c not in ["date", "MADRS"]]
neural_cols = [c for c in df_neural_r.columns if c not in ["date", "MADRS"]]

modality_setups = {
    "Audio": (
        df_audio,
        {"Audio": audio_cols},
    ),
    "Video": (
        df_video,
        {"Video": video_cols},
    ),
    "Neural": (
        df_neural_r,
        {"Neural": neural_cols},
    ),
}


df_neural_audio = pd.merge(df_neural_r, df_audio.drop(columns=["MADRS"]), on="date", how="inner")
df_neural_video = pd.merge(df_neural_r, df_video.drop(columns=["MADRS"]), on="date", how="inner")
df_neural_audio_video = pd.merge(df_neural_audio, df_video.drop(columns=["MADRS"]), on="date", how="inner")

modality_setups["Neural + Audio"] = (
    df_neural_audio,
    {
        "Neural": available_features(df_neural_audio, neural_cols),
        "Audio": available_features(df_neural_audio, audio_cols),
    },
)
modality_setups["Neural + Video"] = (
    df_neural_video,
    {
        "Neural": available_features(df_neural_video, neural_cols),
        "Video": available_features(df_neural_video, video_cols),
    },
)
modality_setups["Neural + Audio + Video"] = (
    df_neural_audio_video,
    {
        "Neural": available_features(df_neural_audio_video, neural_cols),
        "Audio": available_features(df_neural_audio_video, audio_cols),
        "Video": available_features(df_neural_audio_video, video_cols),
    },
)


metrics_by_condition = {}
best_result = {
    "r2": -np.inf,
    "modality": None,
    "use_within_modality_pca": None,
    "pred_df": None,
}

for use_within_modality_pca in [False, True]:
    metrics_rows = []
    for modality_name, (modality_df, modality_feature_map) in modality_setups.items():
        df_pred, corr, r2_, mae, mse = run_loso(
            modality_df,
            target_col="MADRS",
            modality_feature_map=modality_feature_map,
            use_within_modality_pca=use_within_modality_pca,
            n_components=2,
        )

        if r2_ > best_result["r2"]:
            best_result = {
                "r2": r2_,
                "modality": modality_name,
                "use_within_modality_pca": use_within_modality_pca,
                "pred_df": df_pred.copy(),
            }

        metrics_rows.append(
            {
                "Modality": modality_name,
                "Correlation": corr,
                "R2": r2_,
                "MAE": mae,
                "MSE": mse,
            }
        )

    metrics_by_condition[use_within_modality_pca] = pd.DataFrame(metrics_rows).set_index("Modality")


df_metrics_combined = pd.concat(
    {
        "WithinModalityPCA=False": metrics_by_condition[False],
        "WithinModalityPCA=True": metrics_by_condition[True],
    },
    names=["Condition"],
).reset_index()

df_metrics_combined.to_csv(
    "case-report-decoding-comparison/decoding_metrics_across_modalities_neural_within_modality_pca.csv",
    index=False,
)

fig, axes = plt.subplots(2, 4, figsize=(16, 10), sharex=False)
metric_names = ["Correlation", "R2", "MAE", "MSE"]

for row_idx, use_within_modality_pca in enumerate([False, True]):
    df_metrics = metrics_by_condition[use_within_modality_pca]
    for col_idx, metric in enumerate(metric_names):
        ax = axes[row_idx, col_idx]
        sns.barplot(
            x=df_metrics.index,
            y=df_metrics[metric],
            color="#4C78A8",
            ax=ax,
        )
        ax.set_title(f"{metric} | WithinModalityPCA={use_within_modality_pca}")
        ax.tick_params(axis="x", rotation=90)
        ax.set_ylabel(metric)
        if metric in ["Correlation", "R2"]:
            ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig(
    "case-report-decoding-comparison/decoding_metrics_across_modalities_neural_within_modality_pca.pdf"
)


# Plot the best-performing modality/condition by R2.
best_df = best_result["pred_df"].sort_values("date").reset_index(drop=True)

fig_best, (ax_time, ax_reg) = plt.subplots(1, 2, figsize=(14, 5))

ax_time.plot(best_df["date"], best_df["MADRS"], marker="o", label="True MADRS")
ax_time.plot(best_df["date"], best_df["MADRS_pred"], marker="o", label="Predicted MADRS")
ax_time.set_xlabel("Time")
ax_time.set_ylabel("MADRS")
ax_time.set_title(
    f"Best Model by R2: {best_result['modality']} | WithinModalityPCA={best_result['use_within_modality_pca']}"
)
ax_time.tick_params(axis="x", rotation=45)
ax_time.legend()

sns.regplot(
    data=best_df,
    x="MADRS",
    y="MADRS_pred",
    scatter_kws={"s": 60, "alpha": 0.8},
    line_kws={"color": "#F58518", "lw": 2},
    ax=ax_reg,
)

min_val = min(best_df["MADRS"].min(), best_df["MADRS_pred"].min())
max_val = max(best_df["MADRS"].max(), best_df["MADRS_pred"].max())
ax_reg.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="gray", label="Identity")
ax_reg.set_xlabel("True MADRS")
ax_reg.set_ylabel("Predicted MADRS")
ax_reg.set_title(f"Regression (R2={best_result['r2']:.3f})")
ax_reg.legend()

plt.tight_layout()
plt.savefig(
    "case-report-decoding-comparison/best_model_timeseries_and_regression_within_modality_pca.pdf"
)
