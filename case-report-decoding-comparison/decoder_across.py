import pandas as pd
import os
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model, ensemble
import numpy as np
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.svm import SVR


# SELECTED Audio features form Mattson
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

# Mean FAU's per session
df_video_allFAU = pd.read_csv("case-report-decoding-comparison/mean_fau_features_per_date.csv")
df_video_allFAU["date"] = pd.to_datetime(df_video_allFAU["date"], format="%Y-%m-%d")
# drop MADRS column from df_video_allFAU
df_video_allFAU = df_video_allFAU.drop(columns=["MADRS"])

# Mean Audio features per session
df_audio = pd.read_csv("case-report-decoding-comparison/mean_audio_features_per_date.csv")
df_audio["date"] = pd.to_datetime(df_audio["date"], format="%Y-%m-%d")

# Grace computed time-spent in kmeans FAU cluster
df_video = pd.read_csv("/scratch/timonmerk/get_data_NBU/perceptparser/case-report-decoding-comparison/video_loocv_cluster_prop_date_units.csv")
df_video = df_video.drop(columns=["n_csv_in_date", "n_frames_total", "env", "unit_id"])
df_video["date"] = pd.to_datetime(df_video["date"], format="%Y%m%d")

# potentially merge mean FAU and time-spent in kmenas FAU cluster features
# merge df_video_allFAU and df_video on date
#df_video = pd.merge(df_video, df_video_allFAU, on="date", how="inner")

df_neural = pd.read_csv("psd_specparam_features_0_40Hz.csv")
df_neural["date"] = pd.to_datetime(df_neural["date"], format="%Y_%m_%d")
df_neural = df_neural.rename(columns={"madrs": "MADRS"})
df_neural_r = df_neural[df_neural["hemisphere"].str.lower() == "right"].copy()
df_neural_r = df_neural_r.drop(columns=["hemisphere", "nbu_visit", "channel_name"])
#neural_features_use = ["aperiodic_offset", "aperiodic_exponent", "ap_theta", "ap_alpha", "ap_low_beta", "ap_high_beta", ]
#df_neural_r = df_neural_r[["date", "MADRS"] + neural_features_use].copy()

def run_loso(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    use_pca: bool = False,
    n_components: int = 2,
) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True)
    X = df[feature_cols].to_numpy()
    y = df[target_col].to_numpy(dtype=float)

    y_pred = []
    for i in range(len(df)):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i, axis=0)
        X_test = X[i].reshape(1, -1)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        X_train_use = X_train_scaled
        X_test_use = X_test_scaled
        if use_pca:
            n_components_use = min(n_components, X_train_scaled.shape[1], X_train_scaled.shape[0])
            pca = PCA(n_components=n_components_use)
            X_train_use = pca.fit_transform(X_train_scaled)
            X_test_use = pca.transform(X_test_scaled)

        model = linear_model.Ridge(alpha=1.0)
        model = linear_model.LinearRegression()
        model = linear_model.Lasso(alpha=0.1)
        #model = ensemble.RandomForestRegressor(n_estimators=10, random_state=42)
        #model = SVR(kernel="rbf", C=1.0, epsilon=0.1)
        model.fit(X_train_use, y_train)
        y_pred.append(model.predict(X_test_use)[0])

    df_pred = df.copy()
    df_pred["MADRS_pred"] = y_pred
    # clip predictions to be non-negative
    df_pred["MADRS_pred"] = df_pred["MADRS_pred"].clip(lower=0)

    corr = df_pred[[target_col, "MADRS_pred"]].corr().iloc[0, 1]
    r2_ = metrics.r2_score(df_pred[target_col], df_pred["MADRS_pred"])
    mae = metrics.mean_absolute_error(df_pred[target_col], df_pred["MADRS_pred"])
    mse = metrics.mean_squared_error(df_pred[target_col], df_pred["MADRS_pred"])
    return df_pred, corr, r2_, mae, mse

modality_dfs = {
    "Audio": df_audio,
    "Video": df_video,
    "Neural": df_neural_r,
}

# add now combined neural + audio, neural + video, neural + audio + video
df_neural_audio = pd.merge(df_neural_r, df_audio.drop(columns=["MADRS"]), on="date", how="inner")
df_neural_video = pd.merge(df_neural_r, df_video.drop(columns=["MADRS"]), on="date", how="inner")
df_neural_audio_video = pd.merge(df_neural_audio, df_video.drop(columns=["MADRS"]), on="date", how="inner")

modality_dfs["Neural + Audio"] = df_neural_audio
modality_dfs["Neural + Video"] = df_neural_video
modality_dfs["Neural + Audio + Video"] = df_neural_audio_video

metrics_by_pca = {}
best_result = {
    "r2": -np.inf,
    "modality": None,
    "use_pca": None,
    "pred_df": None,
}

for use_pca in [False, True]:
    metrics_rows = []
    for modality_name, modality_df in modality_dfs.items():
        df_pred, corr, r2_, mae, mse = run_loso(
            modality_df,
            modality_df.columns.difference(["date", "MADRS"]),
            "MADRS",
            use_pca=use_pca,
            n_components=2,
        )

        if r2_ > best_result["r2"]:
            best_result = {
                "r2": r2_,
                "modality": modality_name,
                "use_pca": use_pca,
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
    metrics_by_pca[use_pca] = pd.DataFrame(metrics_rows).set_index("Modality")

df_metrics_combined = pd.concat(
    {
        "PCA=False": metrics_by_pca[False],
        "PCA=True": metrics_by_pca[True],
    },
    names=["Condition"],
).reset_index()
df_metrics_combined.to_csv(
    "case-report-decoding-comparison/decoding_metrics_across_modalities_neural_limited.csv",
    index=False,
)

fig, axes = plt.subplots(2, 4, figsize=(16, 10), sharex=False)
metric_names = ["Correlation", "R2", "MAE", "MSE"]

for row_idx, use_pca in enumerate([False, True]):
    df_metrics = metrics_by_pca[use_pca]
    for col_idx, metric in enumerate(metric_names):
        ax = axes[row_idx, col_idx]
        sns.barplot(
            x=df_metrics.index,
            y=df_metrics[metric],
            color="#4C78A8",
            ax=ax,
        )
        ax.set_title(f"{metric} | PCA={use_pca}")
        ax.tick_params(axis="x", rotation=90)
        ax.set_ylabel(metric)
        if metric == "Correlation" or metric == "R2":
            ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig("case-report-decoding-comparison/decoding_metrics_across_modalities_neural_limited.pdf")


# Plot the best-performing setup by R2.
best_df = best_result["pred_df"].sort_values("date").reset_index(drop=True)

fig_best, (ax_time, ax_reg) = plt.subplots(1, 2, figsize=(14, 5))

ax_time.plot(best_df["date"], best_df["MADRS"], marker="o", label="True MADRS")
ax_time.plot(best_df["date"], best_df["MADRS_pred"], marker="o", label="Predicted MADRS")
ax_time.set_xlabel("Time")
ax_time.set_ylabel("MADRS")
ax_time.set_title(f"Best Model by R2: {best_result['modality']} | PCA={best_result['use_pca']}")
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
plt.savefig("case-report-decoding-comparison/best_model_timeseries_and_regression.pdf")