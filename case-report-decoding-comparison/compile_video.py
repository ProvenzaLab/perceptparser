import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

PATH_ = "case-report-decoding-comparison/FAUs"
audio_features = ["AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU10", "AU11", "AU12", "AU13", "AU14", "AU15", "AU16", "AU17", "AU18", "AU19", "AU20", "AU22", "AU23", "AU24", "AU25", "AU26", "AU27", "AU32", "AU38", "AU39", "FAU_L1", "FAU_R1", "FAU_L2", "FAU_R2", "FAU_L4", "FAU_R4", "FAU_L6", "FAU_R6", "FAU_L10", "FAU_R10", "FAU_L12", "FAU_R12", "FAU_L14", "FAU_R14"]
files_ = os.listdir(PATH_)
df_scores = pd.read_csv("case-report-decoding-comparison/FAUs/clinical_scores.csv")
# convert date column from int to str
df_scores["date"] = df_scores["date"].astype(str)
df = pd.DataFrame()
for file in files_:
    if file.endswith(".csv") and "clinical_scores" not in file:
        df_ = pd.read_csv(os.path.join(PATH_, file))
        df_ = df_.query("mp_face_detected == True")

        date_str = file[:8]
        date_ = pd.to_datetime(date_str, format="%Y%m%d")
        df_mean = df_[audio_features].mean().to_frame().T
        df_mean["date"] = date_
        score = df_scores.query("date == @date_str")["madrs"].values[0]
        df_mean["MADRS"] = score
        df = pd.concat([df, df_mean], axis=0)
    
df.to_csv("case-report-decoding-comparison/mean_fau_features_per_date.csv", index=False)

# plot the top 10 positive and top 10 negative correlations with MADRS
correlations = df.corr()["MADRS"].drop("MADRS")
correlations_sorted = correlations.sort_values(ascending=False)
top_10_pos = correlations_sorted[correlations_sorted > 0].head(10)
top_10_neg = correlations_sorted[correlations_sorted < 0].head(10)
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
sns.barplot(x=top_10_pos.values, y=top_10_pos.index, palette="Blues_d")
plt.title("Positive")
plt.xlabel("Correlation Coefficient")
plt.subplot(1, 2, 2)
sns.barplot(x=top_10_neg.values, y=top_10_neg.index, palette="Reds_d")
plt.title("Negative")
plt.xlabel("Correlation Coefficient")
plt.suptitle("Top 10 Correlations with MADRS (FAU Features)")
plt.tight_layout()
plt.savefig("case-report-decoding-comparison/top_10_correlations_madrs_fau.pdf")