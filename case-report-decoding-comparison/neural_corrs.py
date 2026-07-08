import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def main() -> None:
    df_neural = pd.read_csv("psd_specparam_features_0_40Hz.csv")
    df_neural["date"] = pd.to_datetime(df_neural["date"], format="%Y_%m_%d")
    df_neural = df_neural.rename(columns={"madrs": "MADRS"})

    df_neural_r = df_neural[df_neural["hemisphere"].str.lower() == "right"].copy()

    feature_cols = [
        col
        for col in df_neural_r.columns
        if col not in {"date", "MADRS", "hemisphere", "nbu_visit", "channel_name"}
    ]

    correlations = df_neural_r[feature_cols + ["MADRS"]].corr(numeric_only=True)["MADRS"].drop("MADRS")

    top_10_pos = correlations[correlations > 0].sort_values(ascending=False).head(10)
    top_10_neg = correlations[correlations < 0].sort_values(ascending=True).head(10)

    fig, axes = plt.subplots(1, 2, figsize=(12, 7), sharex=True)

    sns.barplot(x=top_10_pos.values, y=top_10_pos.index, color="#4C78A8", ax=axes[0])
    axes[0].set_title("Top 10 Positive")
    axes[0].set_xlabel("Correlation Coefficient")
    axes[0].set_ylabel("Feature")
    if not top_10_pos.empty:
        axes[0].set_xlim(0, max(top_10_pos.max(), 0))

    sns.barplot(x=top_10_neg.values, y=top_10_neg.index, color="#E45756", ax=axes[1])
    axes[1].set_title("Top 10 Negative")
    axes[1].set_xlabel("Correlation Coefficient")
    axes[1].set_ylabel("")
    if not top_10_neg.empty:
        axes[1].set_xlim(min(top_10_neg.min(), 0), 0)

    fig.suptitle("Top Neural Feature Correlations with MADRS (Right Hemisphere)")
    fig.tight_layout()
    fig.savefig("case-report-decoding-comparison/top_10_correlations_madrs_neural.pdf")


if __name__ == "__main__":
    main()