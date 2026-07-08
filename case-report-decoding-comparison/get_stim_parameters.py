import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

df = pd.read_parquet("/scratch/timonmerk/get_data_NBU/perceptparser/trbd_chronic_df_stim_parameters.parq")

# there are columns "left_frequency", "left_pulse_width", "stim_left"
df["teed_left"] = 1000 * df["left_frequency"] * (df["left_pulse_width"] * (10 **-6)) * ((df["stim_left"] * 10 **-3) ** 2)
df["teed_right"] = 1000 * df["right_frequency"] * (df["right_pulse_width"] * (10 **-6)) * ((df["stim_right"] * 10 **-3) ** 2)

# there is a column "timestamp"
# get only entries where teed_right changes
# dropna based on teed_right
df_r_changes = df[df["teed_right"].diff() != 0].dropna(subset=["teed_right"])
# sort by timestamp
df_r_changes = df_r_changes.sort_values(by="timestamp")

line_width = 0.5
plt.figure(figsize=(15, 15))
plt.subplot(4, 1, 1)
plt.plot(df_r_changes["timestamp"], df_r_changes["teed_right"], linewidth=line_width, linestyle="", marker="o", markersize=2)
plt.title("TEED Right")
plt.legend()
plt.subplot(4, 1, 2)
plt.plot(df_r_changes["timestamp"], df_r_changes["right_frequency"], linewidth=line_width, linestyle="", marker="o", markersize=2)
plt.title("Right Frequency")
plt.legend()
plt.subplot(4, 1, 3)
plt.plot(df_r_changes["timestamp"], df_r_changes["right_pulse_width"], linewidth=line_width, linestyle="", marker="o", markersize=2)
plt.title("Right Pulse Width")
plt.legend()
plt.subplot(4, 1, 4)
plt.plot(df_r_changes["timestamp"], df_r_changes["stim_right"], linewidth=line_width, linestyle="", marker="o", markersize=2)
plt.title("Stim Right")
plt.savefig("stim_changes/teed_right_changes.pdf")

time_period_plot_start = pd.to_datetime("2026-01-01 00:00:00", utc=True)
time_period_plot_end = pd.to_datetime("2026-01-05 23:59:59", utc=True)
df_r_changes_period = df_r_changes[(df_r_changes["timestamp"] >= time_period_plot_start) & (df_r_changes["timestamp"] <= time_period_plot_end)]
plt.figure(figsize=(15, 15))
plt.subplot(4, 1, 1)
plt.plot(df_r_changes_period["timestamp"], df_r_changes_period["teed_right"], linewidth=line_width, linestyle="", marker="o", markersize=2)
plt.title("TEED Right")
plt.subplot(4, 1, 2)
plt.plot(df_r_changes_period["timestamp"], df_r_changes_period["right_frequency"], linewidth=line_width, linestyle="", marker="o", markersize=2)
plt.title("Right Frequency")
plt.subplot(4, 1, 3)
plt.plot(df_r_changes_period["timestamp"], df_r_changes_period["right_pulse_width"], linewidth=line_width, linestyle="", marker="o", markersize=2)
plt.title("Right Pulse Width")
plt.subplot(4, 1, 4)
plt.plot(df_r_changes_period["timestamp"], df_r_changes_period["stim_right"], linewidth=line_width, linestyle="", marker="o", markersize=2)
plt.title("Stim Right")
plt.savefig("stim_changes/teed_right_changes_period.pdf")


# take the moving average of 1day for teed_right, right_frequency, right_pulse_width, stim_right
df_r_changes["teed_right_ma"] = df_r_changes["teed_right"].rolling(window="1D").mean()
df_r_changes["right_frequency_ma"] = df_r_changes["right_frequency"].rolling(window="1D").mean()
df_r_changes["right_pulse_width_ma"] = df_r_changes["right_pulse_width"].rolling(window="1D").mean()
df_r_changes["stim_right_ma"] = df_r_changes["stim_right"].rolling(window="1D").mean()

plt.figure(figsize=(15, 15))
plt.subplot(4, 1, 1)
plt.plot(df_r_changes["timestamp"], df_r_changes["teed_right_ma"], linewidth=line_width, linestyle="", marker="o", markersize=2)
plt.title("TEED Right Moving Average (1 Day)")
plt.subplot(4, 1, 2)
plt.plot(df_r_changes["timestamp"], df_r_changes["right_frequency_ma"], linewidth=line_width, linestyle="", marker="o", markersize=2)
plt.title("Right Frequency Moving Average (1 Day)")
plt.subplot(4, 1, 3)
plt.plot(df_r_changes["timestamp"], df_r_changes["right_pulse_width_ma"], linewidth=line_width, linestyle="", marker="o", markersize=2)
plt.title("Right Pulse Width Moving Average (1 Day)")
plt.subplot(4, 1, 4)
plt.plot(df_r_changes["timestamp"], df_r_changes["stim_right_ma"], linewidth=line_width, linestyle="", marker="o", markersize=2)
plt.title("Stim Right Moving Average (1 Day)")
plt.savefig("stim_changes/teed_right_changes_moving_average.pdf")

# get the mean number of changes per day, count
df_r_changes["date"] = df_r_changes["timestamp"].dt.date
changes_per_day = df_r_changes.groupby("date").size()
mean_changes_per_day = changes_per_day.mean()