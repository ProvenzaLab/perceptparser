from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from scipy import signal
import random
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages
import mne


def brain_sense_lfp_plot(df: pd.DataFrame, parser, out_path: str, quantile: float = 0.99):
    df = df
    parser = parser
    out_path = out_path

    plt.figure(figsize=(15, 8))
    plt.subplot(3, 1, 1)
    plt.plot(df.index, df["Left_power"], label="Left LFP")
    plt.plot(df.index, df["Right_power"], label="Right LFP")
    plt.xlabel("Time")
    plt.ylabel("LFP Power [a.u.]")
    plt.title(f"LFP power over Time, date: {parser.session_date.strftime('%Y-%m-%d %H:%M:%S')} - {parser.lead_location} - {parser.lead_model}\n"
            f"{df.index[0].strftime('%Y-%m-%d %H:%M:%S')} - {df.index[-1].strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"LeadLocation: {parser.lead_location}, LeadModel: {parser.lead_model}\n"
            f"All samples")

    plt.subplot(3, 1, 2)
    plt.plot(df.index, df["Left_power"], label="Left LFP", alpha=0.8)
    plt.plot(df.index, df["Right_power"], label="Right LFP", alpha=0.8)
    plt.xlabel("Time")
    plt.ylabel("LFP power [a.u.]")
    plt.title(f"All samples - clipped 0.999 quantile")
    y_max = df[["Left_power", "Right_power"]].quantile(quantile).max()
    plt.ylim(0, y_max)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(df.index, df["Left_power"], label="Left LFP", alpha=0.8)
    plt.plot(df.index, df["Right_power"], label="Right LFP", alpha=0.8)
    plt.xlabel("Time")
    plt.ylabel("LFP power [a.u.]")
    plt.title(f"First 1000 samples")
    y_max = df[["Left_power", "Right_power"]].quantile(quantile).max()
    plt.ylim(0, y_max)
    plt.xlim(df.index[0], df.index[1000])
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_path}/lfp_power_plot.pdf")
    plt.close()

def lfptrendlog_plot(df_trend_logs: pd.DataFrame, parser, path_out: str):
    df_trend_logs = df_trend_logs
    parser = parser
    path_out = path_out

    plt.figure(figsize=(15, 5))
    for hem in df_trend_logs["Hemisphere"].unique():
        df_hem = df_trend_logs[df_trend_logs["Hemisphere"] == hem]
        plt.plot(df_hem.index, df_hem["LFP"], label=f"{hem} LFP Trend Logs")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.xlabel("Time")
    plt.ylabel("LFP Trend Power [a.u.]")
    plt.title(f"LFP Trend Logs over Time, date: {parser.session_date.strftime('%Y-%m-%d %H:%M:%S')} - {parser.lead_location} - {parser.lead_model}\n"
            f"{df_trend_logs.index[0].strftime('%Y-%m-%d %H:%M:%S')} - {df_trend_logs.index[-1].strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"LeadLocation: {parser.lead_location}, LeadModel: {parser.lead_model}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{path_out}/lfp_trend_logs_plot.pdf")
    plt.close()

def brainsense_timedomain_plot(df_bs_td: pd.DataFrame, parser, out_path: str, indefinite_streaming: bool = False):
    ch_names = list(df_bs_td.columns)

    plt.figure(figsize=(15, 5))
    for ch_idx, ch in enumerate(ch_names):
        plt.plot(df_bs_td.index, df_bs_td[ch] + 100*ch_idx, label=ch)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title(f"BrainSense Time Domain Data - {parser.session_date.strftime('%Y-%m-%d %H:%M:%S')} - {parser.lead_location} - {parser.lead_model}\n"
            f"{df_bs_td.index[0].strftime('%Y-%m-%d %H:%M:%S')} - {df_bs_td.index[-1].strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"LeadLocation: {parser.lead_location}, LeadModel: {parser.lead_model}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_path}/brainsense_time_domain_plot_total.pdf")
    plt.close()

    plt.figure(figsize=(15, 5))
    samples_plot = 250 * 10
    for ch_idx, ch in enumerate(ch_names):
        plt.plot(df_bs_td.index, df_bs_td[ch] + 100*ch_idx, label=ch)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title(f"BrainSense Time Domain Data - {parser.session_date.strftime('%Y-%m-%d %H:%M:%S')} - {parser.lead_location} - {parser.lead_model}\n"
            f"{df_bs_td.index[0].strftime('%Y-%m-%d %H:%M:%S')} - {df_bs_td.index[-1].strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"LeadLocation: {parser.lead_location}, LeadModel: {parser.lead_model}")
    plt.legend()
    plt.xlim(df_bs_td.index[0], df_bs_td.index[samples_plot])
    y_min = df_bs_td.iloc[:samples_plot].min().min() - 50
    y_max = df_bs_td.iloc[:samples_plot].max().max() + 50 + 100 * (len(ch_names) - 1)
    plt.ylim(y_min, y_max)
    plt.tight_layout()
    plt.savefig(f"{out_path}/brainsense_time_domain_plot_10s.pdf")
    plt.close()

    plt.figure(figsize=(15, 5))
    samples_plot = 250 * 2
    for ch_idx, ch in enumerate(ch_names):
        plt.plot(df_bs_td.index, df_bs_td[ch] + 100*ch_idx, label=ch)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title(f"BrainSense Time Domain Data - {parser.session_date.strftime('%Y-%m-%d %H:%M:%S')} - {parser.lead_location} - {parser.lead_model}\n"
            f"{df_bs_td.index[0].strftime('%Y-%m-%d %H:%M:%S')} - {df_bs_td.index[-1].strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"LeadLocation: {parser.lead_location}, LeadModel: {parser.lead_model}")
    plt.legend()
    # limit to first 1000 samples
    plt.xlim(df_bs_td.index[0], df_bs_td.index[samples_plot])
    # set ylim to min max of all channels in the 1000 sample range
    y_min = df_bs_td.iloc[:samples_plot].min().min() - 50
    y_max = df_bs_td.iloc[:samples_plot].max().max() + 50 + 100 * (len(ch_names) - 1)
    plt.ylim(y_min, y_max)
    plt.tight_layout()
    plt.savefig(f"{out_path}/brainsense_time_domain_plot_2s.pdf")
    plt.close()

    df_bs_td_ = df_bs_td.reset_index()
    groups = df_bs_td_.groupby(pd.Grouper(key='Time', freq='1min'))

    if indefinite_streaming is True:
        plt.figure(figsize=(15, 12))
    else:
        plt.figure(figsize=(15, 5))
    Pxx_chs_ = []
    t_bins_chs_ = []
    for ch_idx, ch in enumerate(ch_names):
        if indefinite_streaming:
            plt.subplot(len(ch_names) // 2, 2 , ch_idx + 1)
        else:
            plt.subplot(1, len(ch_names), ch_idx + 1)
        plt.title(f"{ch} - {parser.session_date.strftime('%Y-%m-%d %H:%M:%S')} - {parser.lead_location} - {parser.lead_model}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("PSD (V²/Hz)")
        Pxx_l = []
        t_bin_l = []
        for time_bin, df_bin in groups:
            data_ch = df_bin[ch].values
            notnan = ~np.isnan(data_ch)
            
            if not notnan.any():
                print(f"{time_bin} {ch}: No valid data")
                # add NaN to Pxx_l and t_bin_l
                Pxx_l.append(np.nan)
                t_bin_l.append(time_bin)
                continue

            # Find contiguous non-NaN segments
            group_id = np.cumsum(~notnan)
            group_id[~notnan] = -1

            valid_groups, counts = np.unique(group_id[group_id != -1], return_counts=True)
            longest_group = valid_groups[np.argmax(counts)]
            data_clean = data_ch[group_id == longest_group]

            if len(data_clean) < 256:
                print(f"{time_bin} {ch}: Not enough data for PSD")
                Pxx_l.append(np.nan)
                t_bin_l.append(time_bin)
                continue

            # Compute PSD
            f, Pxx_den = signal.welch(data_clean, fs=250, nperseg=256)

            plt.plot(f, np.log(Pxx_den), label=ch, color="black", alpha=0.1)
            Pxx_l.append(Pxx_den)
            t_bin_l.append(time_bin)

        # plt mean
        # if Pxx_l:
        #     Pxx_mean = np.mean(Pxx_l, axis=0)
        #     plt.plot(f, np.log(Pxx_mean), label=f"{ch} mean", color='black', linewidth=1)

        plt.title(f"{ch}")
        Pxx_chs_.append(Pxx_l)
        t_bins_chs_.append(t_bin_l)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("PSD (V²/Hz)")
    plt.suptitle(f"BrainSense Time Domain PSD - {parser.session_date.strftime('%Y-%m-%d %H:%M:%S')} - {parser.lead_location} - {parser.lead_model}\n"
                f"{df_bs_td.index[0].strftime('%Y-%m-%d %H:%M:%S')} - {df_bs_td.index[-1].strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"LeadLocation: {parser.lead_location}, LeadModel: {parser.lead_model}")
    plt.tight_layout()
    plt.savefig(f"{out_path}/brainsense_time_domain_psd.pdf")
    plt.close()

def time_frequency_plot_td(df_bs_td: pd.DataFrame, indefinite_streaming: bool, parser, out_path: str):
    
    if "idx_counter" in df_bs_td.columns:
        df_bs_td = df_bs_td.drop(columns=["idx_counter"])

    ch_names = list(df_bs_td.columns)
    
    if indefinite_streaming is False:
        str_prefix = "bstd_"
    else:
        str_prefix = "istd_"

    if indefinite_streaming is True:
        plt.figure(figsize=(15, 15),)
    else:
        plt.figure(figsize=(15, 7),)

    for ch_idx, ch in enumerate(ch_names):
        if indefinite_streaming:
            plt.subplot(len(ch_names) // 2, 2 , ch_idx + 1)
        else:
            plt.subplot(1, len(ch_names), ch_idx + 1)
        
        df_ = df_bs_td[ch]
        # chunk data up in 10 s intervals
        t_bins_10s = pd.date_range(start=df_.index[0], end=df_.index[-1], freq='10S')
        df_chunks = [df_.loc[t_bins_10s[i]:t_bins_10s[i + 1]] for i in range(len(t_bins_10s) - 1)]

        Pxx_chs_ = []
        t_bins_chs_ = []
        for df_chunk in df_chunks:
            if df_chunk.isnull().all():
                print(f"{ch}: No valid data in chunk")
                continue

            # Compute PSD
            f, Pxx_den = signal.welch(df_chunk.fillna(0), fs=250, nperseg=256)

            # Store the PSD and time bin
            Pxx_chs_.append(Pxx_den)
            t_bins_chs_.append(df_chunk.index[0])
        if len(Pxx_chs_) == 0:
            print(f"{ch}: No valid data for PSD")
            continue
        Pxx_arr = np.array(Pxx_chs_)
        Pxx_log = np.log(Pxx_arr + 1e-12)  # To avoid log(0)


        t_num = mdates.date2num(t_bins_chs_)
        extent = [t_num[0], t_num[-1], f[0], f[-1]]

        plt.imshow(Pxx_log.T, aspect='auto', origin='lower',
                    extent=extent,
                    cmap='viridis', interpolation='none')

        plt.colorbar(label='log(PSD)')
        plt.xlabel('Time')
        plt.ylabel('Frequency (Hz)')
        plt.title(f'{ch}')
        plt.clim(-7.5, 4)
        # Format x-axis as dates
        ax = plt.gca()
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.gcf().autofmt_xdate()

    plt.suptitle(f"BrainSense Time Domain TF - {parser.session_date.strftime('%Y-%m-%d %H:%M:%S')} - {parser.lead_location} - {parser.lead_model}\n"
                f"{df_bs_td.index[0].strftime('%Y-%m-%d %H:%M:%S')} - {df_bs_td.index[-1].strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"LeadLocation: {parser.lead_location}, LeadModel: {parser.lead_model}")
    plt.tight_layout()
    start_idx = df_bs_td.index[0]
    end_idx = df_bs_td.index[-1]
    str_postfix = f"{start_idx.strftime('%Y%m%d_%H%M%S')}_{end_idx.strftime('%Y%m%d_%H%M%S')}"
    plt.savefig(f"{out_path}/time_domain_tf_{str_prefix}{str_postfix}.pdf")
    plt.close()

def plot_time_domain_ranges(dfs_: list[pd.DataFrame], out_path: str):

    plt.figure(figsize=(13, 8))

    for df in dfs_:
        start = df.index[0]
        end = df.index[-1]
        label = f"{start.strftime('%Y-%m-%d %H:%M:%S')} → {end.strftime('%Y-%m-%d %H:%M:%S')}"
        color = [random.random() for _ in range(3)]
        
        plt.axvspan(start, end, alpha=0.3, color=color, label=label)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d %H:%M'))

    plt.xlabel("Time (Day Hour:Minute)")
    plt.ylabel("Interval Indicator")
    plt.title("Start-End Index Intervals for Each DataFrame")

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Intervals", fontsize="small")

    plt.tight_layout()
    #plt.show()
    plt.savefig(f"{out_path}/time_domain_ranges.pdf")
    plt.close()

def plot_df_timeseries(df_plt: pd.DataFrame, out_path: str, brain_sense_timedomain: bool = True):
    #df_plt = dfs_[15]
    if brain_sense_timedomain:
        str_prefix = "bstd_"
    else:
        str_prefix = "istd_"
    pdf_path = f"{out_path}/{str_prefix}{df_plt.index[0]}_{df_plt.index[-1]}.pdf"
    pdf_ = PdfPages(pdf_path)
    TIME_INTERVAL = 10
    samples = 250 * TIME_INTERVAL
    fs = 250
    FILTER = False
    df_plt["idx_counter"] = np.arange(df_plt.shape[0]) // samples
    chs = df_plt.columns[0:2]  # Exclude 'timestamp' and 'idx_counter'
    for idx_cnt in tqdm(df_plt["idx_counter"].unique()):
        df_range_ = df_plt[df_plt["idx_counter"] == idx_cnt]

        plt.figure(figsize=(12, 5))
        plt.suptitle(f"Time Series - idx: {idx_cnt}\n" +
                        f"ch1: {chs[0]} ch2: {chs[1]}\n" +
                        f"{df_range_.index[0]} - {df_range_.index[-1]}")
        for ch_idx, ch_name in enumerate(chs):
            data_ = df_range_[ch_name].to_numpy()
            # fill nans with 0
            if FILTER:
                data_raw = np.nan_to_num(data_, nan=0.0)
                data_filtered = mne.filter.filter_data(
                    data_raw,
                    sfreq=fs,
                    l_freq=105,
                    h_freq=95,
                    method='iir',
                    verbose=False
                )
                data_filtered = mne.filter.filter_data(
                    data_filtered,
                    sfreq=fs,
                    l_freq=65,
                    h_freq=55,
                    method='iir',
                    verbose=False
                )
                data_filtered = mne.filter.filter_data(
                    data_filtered,
                    sfreq=fs,
                    l_freq=0.5,
                    h_freq=None,
                    method='iir',
                    verbose=False
                )
                data_ = data_filtered

            plt.subplot(2, 3, ch_idx*3 + 1)
            plt.plot(np.arange(0, data_.shape[0]/fs, 1/fs), data_, linewidth=0.5)
            plt.gca().spines['right'].set_visible(False); plt.gca().spines['top'].set_visible(False)
            plt.xlabel("Time [s]")
            plt.ylabel("Amplitude [a.u.]")
            plt.subplot(2, 3, ch_idx*3 + 2)
            plt.plot(np.arange(0, data_[:250].shape[0]/fs, 1/fs), data_[:250], linewidth=0.5)
            plt.gca().spines['right'].set_visible(False); plt.gca().spines['top'].set_visible(False)
            plt.xlabel("Time [s]")
            plt.ylabel("Amplitude [a.u.]")
            plt.subplot(2, 3, ch_idx*3 + 3)
            if FILTER is False:
                data_ = np.nan_to_num(data_, nan=0.0)
            f, Pxx = signal.welch(data_, fs=fs, nperseg=250)
            plt.plot(f, np.log(Pxx), linewidth=0.5)
            plt.gca().spines['right'].set_visible(False); plt.gca().spines['top'].set_visible(False)
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("PSD [a.u.]")
        plt.tight_layout()
        pdf_.savefig(bbox_inches='tight')
        plt.close()
    pdf_.close()
