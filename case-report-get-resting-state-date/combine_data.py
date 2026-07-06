import pandas as pd
import os
from scipy import signal
from matplotlib import pyplot as plt
import numpy as np 
from mne.time_frequency import psd_array_multitaper
from tqdm import tqdm


df_segments_NBU = pd.read_parquet("TRBD001_selected_NBU.parq")

# query for date
d_visits = {}
d_visits["2025_06_03"] = df_segments_NBU.loc["2025-06-03"]
d_visits["2025_06_04"] = df_segments_NBU.loc["2025-06-04"]
d_visits["2025_06_10"] = df_segments_NBU.loc["2025-06-10"]
d_visits["2025_06_11"] = pd.read_csv("/scratch/timonmerk/get_data_NBU/perceptparser/Jamail_rs_trbd001/Report_Json_Session_Report_20250611T110521/BrainSenseTimeDomain_2025-06-11_13-30-48_13-35-13.csv")
d_visits["2025_08_12"] = pd.read_csv("/scratch/timonmerk/get_data_NBU/perceptparser/Jamail_rs_trbd001/Report_Json_Session_Report_20250812T134246/BrainSenseTimeDomain_2025-08-12_19-41-55_19-43-21.csv")
d_visits["2025_10_07"] = pd.read_csv("/scratch/timonmerk/get_data_NBU/perceptparser/Jamail_rs_trbd001/Report_Json_Session_Report_20251008T122314/BrainSenseTimeDomain_2025-10-07_18-40-47_18-42-29.csv")
d_visits["2025_11_04"] = df_segments_NBU.loc["2025-11-04"]
d_visits["2025_12_04"] = pd.read_csv("/scratch/timonmerk/get_data_NBU/perceptparser/Jamail_rs_trbd001/Report_Json_Session_Report_20251204T143810/BrainSenseTimeDomain_2025-12-04_17-56-34_17-59-37.csv")
d_visits["2026_01_21"] = pd.read_csv("/scratch/timonmerk/get_data_NBU/perceptparser/Jamail_rs_trbd001/Report_Json_Session_Report_20260121T141220/BrainSenseTimeDomain_2026-01-21_16-20-47_16-23-52.csv")
d_visits["2026_02_19"] = df_segments_NBU.loc["2026-02-19"]
d_visits["2026_03_26"] = pd.read_csv("/scratch/timonmerk/get_data_NBU/perceptparser/Jamail_rs_trbd001/Report_Json_Session_Report_20260326T140456/BrainSenseTimeDomain_2026-03-26_14-09-37_14-12-38.csv")
d_visits["2026_04_16"] = pd.read_csv("/scratch/timonmerk/get_data_NBU/perceptparser/Jamail_rs_trbd001/Report_Json_Session_Report_20260416T114950/BrainSenseTimeDomain_2026-04-16_15-43-29_15-46-33.csv")
d_visits["2026_05_12"] = df_segments_NBU.loc["2026-05-12"]
d_visits["2026_06_10"] = pd.read_csv("/scratch/timonmerk/get_data_NBU/perceptparser/Jamail_rs_trbd001/Report_Json_Session_Report_20260610T112955/BrainSenseTimeDomain_2026-06-10_15-33-18_15-36-24.csv")
d_visits["2026_06_15"] = pd.read_csv("/scratch/timonmerk/get_data_NBU/perceptparser/Jamail_rs_trbd001/Report_Json_Session_Report_20260615T132557/BrainSenseTimeDomain_2026-06-15_16-09-56_16-13-13.csv")

NBU_visits = {
    "2025_06_03" : True,
    "2025_06_04" : True,
    "2025_06_10" : True,
    "2025_06_11" : True,
    "2025_08_12" : False,
    "2025_10_07" : False,
    "2025_11_04" : True,
    "2025_12_04" : False,
    "2026_01_21" : False,
    "2026_02_19" : False,
    "2026_03_26" : False,
    "2026_04_16" : False,
    "2026_05_12" : True,
    "2026_06_10" : False,
    "2026_06_15" : False,
}

madrs = {
    "2025_04_16": 39,
    "2025_06_03": 29,
    "2025_06_04": 29,
    "2025_06_10": 23,
    "2025_06_11": 23,
    "2025_06_17": 14,
    "2025_06_18": 14,
    "2025_07_17": 20,
    "2025_08_12": 12,
    "2025_09_11": 5,
    "2025_10_07": 6,
    "2025_11_04": 2,
    "2025_12_04": 2,
    "2026_01_21": 8,
    "2026_02_19": 2,
    "2026_03_26": 0,
    "2026_04_16": 0,
    "2026_05_12": 0,
    "2026_06_10": 4,
    "2026_06_15": 1,
}

d_left = {}
d_right = {}
channel_name_left = {}
channel_name_right = {}
channels_left = []
channels_right = []
for date, df in d_visits.items():
    # check if df has column "Channel"
    # if so, get the unique Channels
    # iterate through them if LEFT or RIGHT, and store in d_left and d_right
    # if df doesn't have Channel, iterate through the columns and check if "LEFT" or "RIGHT" is in the column name, and store in d_left and d_right
    if "Channel" in df.columns:
        for channel in df["Channel"].unique():
            if "LEFT" in channel:
                # set "Time" to be index
                if "Time" in df.columns:
                    df = df.set_index("Time")
                df_add = df[df["Channel"] == channel]
                df_add = df_add.rename({"Value": f"{channel}"}, axis=1)
                d_left[date] = df_add[[f"{channel}"]]
                channel_name_left[date] = channel
                channels_left.append(f"{channel}")
            elif "RIGHT" in channel:
                if "Time" in df.columns:
                    df = df.set_index("Time")
                df_add = df[df["Channel"] == channel]
                df_add = df_add.rename({"Value": f"{channel}"}, axis=1)
                d_right[date] = df_add[[f"{channel}"]]
                channel_name_right[date] = channel
                channels_right.append(f"{channel}")
    else:
        for col in df.columns:
            if "LEFT" in col and not df[col].isnull().all(): # if not all nan
                d_left[date] = df[[col]]
                channel_name_left[date] = col
                channels_left.append(f"{col}")
            elif "RIGHT" in col and not df[col].isnull().all():
                d_right[date] = df[[col]]
                channel_name_right[date] = col
                channels_right.append(f"{col}")

# now for each date, compute psd
psds_left = {}
psds_right = {}

for date, df in tqdm(d_left.items()):
    # compute psd, use pwelch, fs=250, nperseg=1024, return_onesided=True
    ts = df.values[0:250*120][:, 0]  # first minute of data
    ts_no_nan = ts[~np.isnan(ts)]
    Pxx, f = psd_array_multitaper(
        ts_no_nan,
        sfreq=250,
        bandwidth=2,        # Hz
        adaptive=False,
        normalization="full",
        n_jobs=-1
    )
    # plt.figure()
    # plt.semilogy(f, Pxx)
    # plt.title(f"Left Hemisphere PSD - {date}")
    # plt.xlabel("Frequency [Hz]")
    # plt.ylabel("PSD [V**2/Hz]")
    # plt.xlim([0, 40])
    # plt.savefig(f"psd_left_{date}.pdf")

    psds_left[date] = (f, Pxx)
    ts = d_right[date].values[:250*120][:, 0]
    ts_no_nan = ts[~np.isnan(ts)]
    Pxx, f = psd_array_multitaper(
        ts_no_nan,
        sfreq=250,
        bandwidth=2,        # Hz
        adaptive=False,
        normalization="full",
        n_jobs=-1
    )
    psds_right[date] = (f, Pxx)

# plot each psd, in separate left and right plots, log transformed, color-code by date

for FULL in [False, True]:
    for RANDOM in [False, True]:
        if RANDOM is False:
            colors = plt.cm.viridis(np.linspace(0, 1, len(psds_left)))
        else:
            colors = plt.cm.tab20(np.linspace(0, 1, len(psds_left)))

        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        for date, (f, Pxx) in psds_left.items():
            plt.semilogy(f, Pxx, label=date, color=colors[list(psds_left.keys()).index(date)])
        plt.title("Left Hemisphere PSD")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("PSD [V**2/Hz]")
        if FULL is False:
            plt.xlim([0, 40])
            plt.ylim([10**-2, 2*10**1])
        else:
            plt.xlim([0, 90])
            plt.ylim([3*10**-3, 2*10**1])

        plt.subplot(1, 2, 2)
        for date, (f, Pxx) in psds_right.items():
            plt.semilogy(f, Pxx, label=date, color=colors[list(psds_right.keys()).index(date)])
        plt.title("Right Hemisphere PSD")
        if FULL is False:
            plt.xlim([0, 40])
            plt.ylim([10**-2, 2*10**1])
        else:
            plt.xlim([0, 90])
            plt.ylim([3*10**-3, 2*10**1])

        plt.xlabel("Frequency [Hz]")
        plt.ylabel("PSD [V**2/Hz]")
        # legend outside of the plot
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout()
        if FULL is False:
            if RANDOM is False:
                plt.savefig("psd_left_right_2min.pdf")
            else:
                plt.savefig("psd_left_right_2min_random.pdf")
        else:
            if RANDOM is False:
                plt.savefig("psd_left_right_2min_full.pdf")
            else:
                plt.savefig("psd_left_right_2min_full_random.pdf")

    # color-code by MADRS score, with a colormap from red (high) to blue (low)
    colors = plt.cm.viridis(np.linspace(0, 1, len(madrs)))
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    for date, (f, Pxx) in psds_left.items():
        madrs_score = madrs[date]
        plt.semilogy(f, Pxx, label=f"{date} (MADRS: {madrs_score})", color=colors[list(madrs.keys()).index(date)])
    plt.title("Left Hemisphere PSD")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD [V**2/Hz]")
    if FULL is False:
        plt.xlim([0, 40])
        plt.ylim([10**-2, 2*10**1])
    else:
        plt.xlim([0, 90])
        plt.ylim([3*10**-3, 2*10**1])

    plt.subplot(1, 2, 2)
    for date, (f, Pxx) in psds_right.items():
        madrs_score = madrs[date]
        plt.semilogy(f, Pxx, label=f"{date} (MADRS: {madrs_score})", color=colors[list(madrs.keys()).index(date)])
    plt.title("Right Hemisphere PSD")
    if FULL is False:
        plt.xlim([0, 40])
        plt.ylim([10**-2, 2*10**1])
    else:
        plt.xlim([0, 90])
        plt.ylim([3*10**-3, 2*10**1])
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD [V**2/Hz]")

    # add a legend with MADRS scores, outside of the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    if FULL is False:
        plt.savefig("psd_left_right_2min_madrs.pdf")
    else:
        plt.savefig("psd_left_right_2min_full_madrs.pdf")
    # define response as 65% of highest MADRS score, which is 29, so 65% of 29 is 18.85, so 19. If MADRS score is below 19, then response is True, else False
    response = {date: (score < 19) for date, score in madrs.items()}

    # plot the response on the psd plots, with a vertical line at 19, and color-code by response
    # color-code blue vs red, blue for response True, red for response False
    colors = {True: "blue", False: "red"}
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    for date, (f, Pxx) in psds_left.items():
        plt.semilogy(f, Pxx, label=f"{date} (MADRS: {madrs[date]})", color=colors[response[date]])
    plt.title("Left Hemisphere PSD")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD [V**2/Hz]")
    if FULL is False:
        plt.xlim([0, 40])
        plt.ylim([10**-2, 2*10**1])
    else:
        plt.xlim([0, 90])
        plt.ylim([3*10**-3, 2*10**1])
    plt.subplot(1, 2, 2)
    for date, (f, Pxx) in psds_right.items():
        plt.semilogy(f, Pxx, label=f"{date} (MADRS: {madrs[date]})", color=colors[response[date]])
    plt.title("Right Hemisphere PSD")
    if FULL is False:
        plt.xlim([0, 40])
        plt.ylim([10**-2, 2*10**1])
    else:
        plt.xlim([0, 90])
        plt.ylim([3*10**-3, 2*10**1])
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD [V**2/Hz]")
    # add a legend with MADRS scores, outside of the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    if FULL is False:
        plt.savefig("psd_left_right_2min_response.pdf")
    else:
        plt.savefig("psd_left_right_2min_full_response.pdf")


    # make another plot with MADRS smaller 10, colors should be binary blue or red if madrs is below 10
    colors = {True: "blue", False: "red"}

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    for date, (f, Pxx) in psds_left.items():
        plt.semilogy(f, Pxx, label=f"{date} (MADRS: {madrs[date]})", color=colors[madrs[date] < 10])
    plt.title("Left Hemisphere PSD")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD [V**2/Hz]")
    if FULL is False:
        plt.xlim([0, 40])
        plt.ylim([10**-2, 2*10**1])
    else:
        plt.xlim([0, 90])
        plt.ylim([3*10**-3, 2*10**1])
    plt.subplot(1, 2, 2)
    for date, (f, Pxx) in psds_right.items():
        plt.semilogy(f, Pxx, label=f"{date} (MADRS: {madrs[date]})", color=colors[madrs[date] < 10])
    plt.title("Right Hemisphere PSD")
    if FULL is False:
        plt.xlim([0, 40])
        plt.ylim([10**-2, 2*10**1])
    else:
        plt.xlim([0, 90])
        plt.ylim([3*10**-3, 2*10**1])
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD [V**2/Hz]")
    # add a legend with MADRS scores, outside of the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    if FULL is False:
        plt.savefig("psd_left_right_2min_small_madrs.pdf")
    else:
        plt.savefig("psd_left_right_2min_full_small_madrs.pdf")


    # color-code by channel name (categorical)
    left_channel_labels = sorted(set(channel_name_left.values()))
    right_channel_labels = sorted(set(channel_name_right.values()))
    left_channel_colors = {
        label: plt.cm.tab10(i / max(1, len(left_channel_labels) - 1))
        for i, label in enumerate(left_channel_labels)
    }
    right_channel_colors = {
        label: plt.cm.tab10(i / max(1, len(right_channel_labels) - 1))
        for i, label in enumerate(right_channel_labels)
    }

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    for date, (f, Pxx) in psds_left.items():
        channel_label = channel_name_left[date]
        plt.semilogy(f, Pxx, label=f"{date} ({channel_label})", color=left_channel_colors[channel_label])
    plt.title("Left Hemisphere PSD (Color by Channel)")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD [V**2/Hz]")
    if FULL is False:
        plt.xlim([0, 40])
        plt.ylim([10**-2, 2*10**1])
    else:
        plt.xlim([0, 90])
        plt.ylim([3*10**-3, 2*10**1])

    plt.subplot(1, 2, 2)
    for date, (f, Pxx) in psds_right.items():
        channel_label = channel_name_right[date]
        plt.semilogy(f, Pxx, label=f"{date} ({channel_label})", color=right_channel_colors[channel_label])
    plt.title("Right Hemisphere PSD (Color by Channel)")
    if FULL is False:
        plt.xlim([0, 40])
        plt.ylim([10**-2, 2*10**1])
    else:
        plt.xlim([0, 90])
        plt.ylim([3*10**-3, 2*10**1])
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD [V**2/Hz]")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    if FULL is False:
        plt.savefig("psd_left_right_2min_by_channel_name.pdf")
    else:
        plt.savefig("psd_left_right_2min_full_by_channel_name.pdf")


    # color-code by NBU visit status (binary)

    nbu_colors = {True: "tab:blue", False: "tab:red"}

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    for date, (f, Pxx) in psds_left.items():
        plt.semilogy(f, Pxx, label=f"{date} (NBU: {NBU_visits[date]})", color=nbu_colors[NBU_visits[date]])
    plt.title("Left Hemisphere PSD (Color by NBU)")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD [V**2/Hz]")
    if FULL is False:
        plt.xlim([0, 40])
        plt.ylim([10**-2, 2*10**1])
    else:
        plt.xlim([0, 90])
        plt.ylim([3*10**-3, 2*10**1])

    plt.subplot(1, 2, 2)
    for date, (f, Pxx) in psds_right.items():
        plt.semilogy(f, Pxx, label=f"{date} (NBU: {NBU_visits[date]})", color=nbu_colors[NBU_visits[date]])
    plt.title("Right Hemisphere PSD (Color by NBU)")
    if FULL is False:
        plt.xlim([0, 40])
        plt.ylim([10**-2, 2*10**1])
    else:
        plt.xlim([0, 90])
        plt.ylim([3*10**-3, 2*10**1])
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD [V**2/Hz]")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    if FULL is False:
        plt.savefig("psd_left_right_2min_by_nbu_status.pdf")
    else:
        plt.savefig("psd_left_right_2min_full_by_nbu_status.pdf")

