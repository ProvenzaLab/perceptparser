import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats

df_visit_times = pd.read_csv("case-report-get-resting-state-date/visit_time_points.csv")

df_impedance_values = pd.read_csv("stim_changes/impedance_values_bipolar.csv")
df_imp_r = df_impedance_values.query("Hemisphere == 'Right'").copy()
# average per SessionDate
df_imp_r_avg = df_imp_r.groupby("SessionDate")["ResultValue"].mean().reset_index()
df_imp_r_avg["SessionDate"] = pd.to_datetime(df_imp_r_avg["SessionDate"])
df_imp_r_avg = df_imp_r_avg.sort_values("SessionDate").reset_index(drop=True)

df_neural = pd.read_csv("psd_specparam_features_0_40Hz.csv")
df_neural_right = df_neural[df_neural["hemisphere"].str.lower() == "right"].copy()
# date has form 2025_04_16, convert to datetime
df_neural_right["date"] = pd.to_datetime(df_neural_right["date"], format="%Y_%m_%d")
df_neural_right = df_neural_right.sort_values("date").reset_index(drop=True)

df_teed = pd.read_parquet("/scratch/timonmerk/get_data_NBU/perceptparser/trbd_chronic_df_stim_parameters.parq")
df_teed["teed_left"] = 1000 * df_teed["left_frequency"] * (df_teed["left_pulse_width"] * (10 **-6)) * ((df_teed["stim_left"] * 10 **-3) ** 2)
df_teed["teed_right"] = 1000 * df_teed["right_frequency"] * (df_teed["right_pulse_width"] * (10 **-6)) * ((df_teed["stim_right"] * 10 **-3) ** 2)

# for each "start_time" in df_visit_times, find the closest previous "timestamp" in df_teed and get the corresponding "teed_right" value
df_visit_times["teed_right"] = df_visit_times["start_time"].apply(
    lambda x: df_teed[df_teed["timestamp"] <= x].sort_values("timestamp", ascending=False).iloc[0]["teed_right"]
)

# find the closest previous "SessionDate" in df_imp_r_avg for each "start_time" in df_visit_times and get the corresponding "ResultValue" (impedance) value
df_visit_times["impedance"] = df_visit_times["start_time"].apply(
    lambda x: df_imp_r_avg[df_imp_r_avg["SessionDate"] <= x].sort_values("SessionDate", ascending=False).iloc[0]["ResultValue"]
)

# merge df_visit_times and df_neural_right based on "date"
df_visit_times["date"] = pd.to_datetime(df_visit_times["date"], format="%Y_%m_%d")
df_merged = pd.merge(df_visit_times, df_neural_right, on="date", how="left")
df_merged["days_since_dbs"] = df_merged["date"] - df_merged["date"].min()
# convert to number of days
df_merged["days_since_dbs"] = df_merged["days_since_dbs"].dt.days

# show all to all correlation in n times n regplots
# use MADRS, impedance, teed_right, ap_high_beta, aperiodic_offset
# plot in the title the correlation coefficient and p-value
features = ["madrs", "impedance", "teed_right", "raw_high_beta", "aperiodic_offset", "days_since_dbs"]
n = len(features)

# compute difference to next value for all features (current - next)
df_plot = df_merged.sort_values("date").copy()
features_diff = []
for feature in features:
    feature_diff = f"{feature}_diff"
    df_plot[feature_diff] = df_plot[feature] - df_plot[feature].shift(-1)
    features_diff.append(feature_diff)


def safe_pearsonr(df, x_col, y_col):
    valid = df[[x_col, y_col]].notna().all(axis=1)
    if valid.sum() < 2:
        return np.nan, np.nan
    return stats.pearsonr(df.loc[valid, x_col], df.loc[valid, y_col])

# show for each of the measure also in barplot the correlations each other
plt.figure(figsize=(10, 10))
for i, feature1 in enumerate(features_diff):
    plt.subplot(3, 2, i + 1)
    corr_coefs = []
    p_values = []
    for feature2 in features_diff:
        if feature1 != feature2:
            corr_coef, p_value = safe_pearsonr(df_plot, feature1, feature2)
            corr_coefs.append(corr_coef)
            p_values.append(p_value)
        else:
            corr_coefs.append(np.nan)
            p_values.append(np.nan)
    ax = sns.barplot(x=features_diff, y=corr_coefs)
    valid_corrs = [corr for corr in corr_coefs if not np.isnan(corr)]
    for bar, corr in zip(ax.patches, valid_corrs):
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()
        y_offset = 0.02 if y >= 0 else -0.02
        va = "bottom" if y >= 0 else "top"
        ax.text(x, y + y_offset, f"{corr:.2f}", ha="center", va=va, fontsize=9)
    plt.title(f"{feature1}_diff")
    plt.ylim(-1.1, 1.1)
    plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("stim_changes/correlations_barplot_all_features_diff.pdf")

plt.figure(figsize=(5*4, 5*4))
for i, feature1 in enumerate(features_diff):
    for j, feature2 in enumerate(features_diff):
        plt.subplot(n, n, i*n + j + 1)
        sns.regplot(x=feature1, y=feature2, data=df_plot)
        corr_coef, p_value = safe_pearsonr(df_plot, feature1, feature2)
        plt.title(f"r={corr_coef:.2f}, p={p_value:.3f}_diff")
plt.tight_layout()
plt.savefig("stim_changes/correlations_all_features_diff.pdf")

# timeline plot with one left axis (madrs) and multiple right axes (other features)
timeline_features = features_diff
timeline_colors = {
    "madrs_diff": "black",
    "impedance_diff": "tab:blue",
    "teed_right_diff": "tab:orange",
    "raw_high_beta_diff": "tab:green",
    "aperiodic_offset_diff": "tab:red",
    "days_since_dbs_diff": "tab:purple",
}

df_timeline = df_plot.sort_values("date").copy()

fig, ax_left = plt.subplots(figsize=(14, 7))

# left axis: MADRS diff
line_left, = ax_left.plot(
    df_timeline["date"],
    df_timeline["madrs_diff"],
    color=timeline_colors["madrs_diff"],
    marker="o",
    linewidth=2,
    label="madrs_diff",
)
ax_left.set_ylabel("madrs_diff", color=timeline_colors["madrs_diff"])
ax_left.tick_params(axis="y", labelcolor=timeline_colors["madrs_diff"])
ax_left.set_xlabel("date")
ax_left.grid(True, axis="x", alpha=0.25)

# right axes: all remaining features, each with its own axis
right_features = [f for f in timeline_features if f != "madrs_diff"]
right_axes = []
right_lines = []

for i, feat in enumerate(right_features):
    ax_r = ax_left.twinx()
    if i > 0:
        ax_r.spines["right"].set_position(("outward", 60 * i))
    color = timeline_colors[feat]
    line_r, = ax_r.plot(
        df_timeline["date"],
        df_timeline[feat],
        color=color,
        marker="o",
        linewidth=2,
        label=feat,
    )
    ax_r.set_ylabel(feat, color=color)
    ax_r.tick_params(axis="y", labelcolor=color)
    right_axes.append(ax_r)
    right_lines.append(line_r)

# combine legend from all axes
all_lines = [line_left] + right_lines
all_labels = [line.get_label() for line in all_lines]
ax_left.legend(all_lines, all_labels, loc="upper left", frameon=True)

plt.title("Timeline of MADRS, impedance, TEED, neural and time features_diff")
fig.subplots_adjust(right=0.78)
plt.tight_layout()
plt.savefig("stim_changes/timeline_multi_axis_features_diff.pdf")
