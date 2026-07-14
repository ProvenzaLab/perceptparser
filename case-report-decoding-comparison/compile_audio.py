import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


df_audio = pd.read_csv("/scratch/timonmerk/whisper_nbu/TRBD001_interview_features_all.csv")
df_audio["date"] = pd.to_datetime(df_audio["date"], format="%m/%d/%Y")
unique_dates = df_audio["date"].unique()
list_audio_features = ["Loudness_sma3", "alphaRatio_sma3", "hammarbergIndex_sma3", "slope0-500_sma3", "slope500-1500_sma3", "spectralFlux_sma3", "mfcc1_sma3", "mfcc2_sma3", "mfcc3_sma3", "mfcc4_sma3", "F0semitoneFrom27.5Hz_sma3nz", "jitterLocal_sma3nz", "shimmerLocaldB_sma3nz", "HNRdBACF_sma3nz", "logRelF0-H1-H2_sma3nz", "logRelF0-H1-A3_sma3nz", "F1frequency_sma3nz", "F1bandwidth_sma3nz", "F1amplitudeLogRelF0_sma3nz", "F2frequency_sma3nz", "F2bandwidth_sma3nz", "F2amplitudeLogRelF0_sma3nz", "F3frequency_sma3nz", "F3bandwidth_sma3nz", "F3amplitudeLogRelF0_sma3nz", "F0semitoneFrom27.5Hz_sma3nz_amean", "F0semitoneFrom27.5Hz_sma3nz_stddevNorm", "F0semitoneFrom27.5Hz_sma3nz_percentile20.0", "F0semitoneFrom27.5Hz_sma3nz_percentile50.0", "F0semitoneFrom27.5Hz_sma3nz_percentile80.0", "F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2", "F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope", "F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope", "F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope", "F0semitoneFrom27.5Hz_sma3nz_stddevFallingSlope", "loudness_sma3_amean", "loudness_sma3_stddevNorm", "loudness_sma3_percentile20.0", "loudness_sma3_percentile50.0", "loudness_sma3_percentile80.0", "loudness_sma3_pctlrange0-2", "loudness_sma3_meanRisingSlope", "loudness_sma3_stddevRisingSlope", "loudness_sma3_meanFallingSlope", "loudness_sma3_stddevFallingSlope", "spectralFlux_sma3_amean", "spectralFlux_sma3_stddevNorm", "mfcc1_sma3_amean", "mfcc1_sma3_stddevNorm", "mfcc2_sma3_amean", "mfcc2_sma3_stddevNorm", "mfcc3_sma3_amean", "mfcc3_sma3_stddevNorm", "mfcc4_sma3_amean", "mfcc4_sma3_stddevNorm", "jitterLocal_sma3nz_amean", "jitterLocal_sma3nz_stddevNorm", "shimmerLocaldB_sma3nz_amean", "shimmerLocaldB_sma3nz_stddevNorm", "HNRdBACF_sma3nz_amean", "HNRdBACF_sma3nz_stddevNorm", "logRelF0-H1-H2_sma3nz_amean", "logRelF0-H1-H2_sma3nz_stddevNorm", "logRelF0-H1-A3_sma3nz_amean", "logRelF0-H1-A3_sma3nz_stddevNorm", "F1frequency_sma3nz_amean", "F1frequency_sma3nz_stddevNorm", "F1bandwidth_sma3nz_amean", "F1bandwidth_sma3nz_stddevNorm", "F1amplitudeLogRelF0_sma3nz_amean", "F1amplitudeLogRelF0_sma3nz_stddevNorm", "F2frequency_sma3nz_amean", "F2frequency_sma3nz_stddevNorm", "F2bandwidth_sma3nz_amean", "F2bandwidth_sma3nz_stddevNorm", "F2amplitudeLogRelF0_sma3nz_amean", "F2amplitudeLogRelF0_sma3nz_stddevNorm", "F3frequency_sma3nz_amean", "F3frequency_sma3nz_stddevNorm", "F3bandwidth_sma3nz_amean", "F3bandwidth_sma3nz_stddevNorm", "F3amplitudeLogRelF0_sma3nz_amean", "F3amplitudeLogRelF0_sma3nz_stddevNorm", "alphaRatioV_sma3nz_amean", "alphaRatioV_sma3nz_stddevNorm", "hammarbergIndexV_sma3nz_amean", "hammarbergIndexV_sma3nz_stddevNorm", "slopeV0-500_sma3nz_amean", "slopeV0-500_sma3nz_stddevNorm", "slopeV500-1500_sma3nz_amean", "slopeV500-1500_sma3nz_stddevNorm", "spectralFluxV_sma3nz_amean", "spectralFluxV_sma3nz_stddevNorm", "mfcc1V_sma3nz_amean", "mfcc1V_sma3nz_stddevNorm", "mfcc2V_sma3nz_amean", "mfcc2V_sma3nz_stddevNorm", "mfcc3V_sma3nz_amean", "mfcc3V_sma3nz_stddevNorm", "mfcc4V_sma3nz_amean", "mfcc4V_sma3nz_stddevNorm", "alphaRatioUV_sma3nz_amean", "hammarbergIndexUV_sma3nz_amean", "slopeUV0-500_sma3nz_amean", "slopeUV500-1500_sma3nz_amean", "spectralFluxUV_sma3nz_amean", "loudnessPeaksPerSec", "VoicedSegmentsPerSec", "MeanVoicedSegmentLengthSec", "StddevVoicedSegmentLengthSec", "MeanUnvoicedSegmentLength", "StddevUnvoicedSegmentLength", "equivalentSoundLevel_dBp", "arousal", "dominance", "valence"]
list_audio_features = [
"F0semitoneFrom27.5Hz_sma3nz_amean",
"F0semitoneFrom27.5Hz_sma3nz_stddevNorm",
"F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2",
"F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope",
"F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope",
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
df_audio_mean = df_audio.groupby("date")[list_audio_features + ["MADRS"]].mean().reset_index()
df_audio_mean.to_csv("case-report-decoding-comparison/mean_audio_features_per_date.csv")
# get each correlations with MADRS
correlations = df_audio_mean.corr()["MADRS"].drop("MADRS")
correlations_sorted = correlations.sort_values(ascending=False)
#correlations_sorted = correlations.abs().sort_values(ascending=False)
# plot those in a pdf
# top 10 pos and top 10 neg correlations
top_10_pos = correlations_sorted[correlations_sorted > 0].head(15)
top_10_neg = correlations_sorted[correlations_sorted < 0].head(15)

plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
sns.barplot(x=top_10_pos.values, y=top_10_pos.index, palette="Blues_d")
plt.title("Positive")
plt.xlabel("Correlation Coefficient")
plt.subplot(1, 2, 2)
sns.barplot(x=top_10_neg.values, y=top_10_neg.index, palette="Reds_d")
plt.title("Negative")
plt.xlabel("Correlation Coefficient")
plt.suptitle("Top 15 Correlations with MADRS (Audio Features)")
plt.tight_layout()
plt.savefig("/scratch/timonmerk/get_data_NBU/perceptparser/case-report-decoding-comparison/top_15_correlations_madrs_audio_limited.pdf")



