import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings

from pathlib import Path
from .stim_settings import FileStimGroupSettings


class PerceptParser:
    def __init__(self, filename: str, verbose: bool = False):
        self.filename = filename
        self.verbose = verbose

        with open(filename, "r") as f:
            self.js = json.load(f)

        self.session_date = pd.Timestamp(self.js["SessionDate"])

        self.lead_location = (
            self.js["LeadConfiguration"]["Final"][0]["LeadLocation"]
            .removeprefix("LeadLocationDef.")
            .upper()
        )

        self.lead_model = (
            self.js["LeadConfiguration"]["Final"][0]["Model"]
            .removeprefix("LeadModelDef.")
            .upper()
        )

        self.subject = self.js["PatientInformation"]["Final"]["PatientId"]
        self.diagnosis = self.js["PatientInformation"]["Final"]["Diagnosis"]

        self.stim_settings = FileStimGroupSettings(self.js, self.filename)
        print(f"{filename}: {self.session_date} - {self.lead_location}")

    def parse_all(self, out_path: str = "sub", plot: bool = False):
        if plot:
            from percept_parser import plotter

        Path.mkdir(Path(out_path), exist_ok=True)

        df_lfp_trend_logs = self.parse_lfp_trend_logs()
        df_brainsense_lfp = self.parse_brain_sense_lfp()

        dfs_bs_td = self.read_timedomain_data(indefinite_streaming=False)
        dfs_is_td = self.read_timedomain_data(indefinite_streaming=True)

        if not df_lfp_trend_logs.empty:
            df_lfp_trend_logs = self._merge_stim_settings(df_lfp_trend_logs)
            df_lfp_trend_logs.to_csv(Path(out_path, "LFPTrendLogs.csv"), index=True)
            if plot:
                plotter.lfptrendlog_plot(df_lfp_trend_logs, self, path_out=out_path)

        if not df_brainsense_lfp.empty:
            df_brainsense_lfp = self._merge_stim_settings(df_brainsense_lfp)
            df_brainsense_lfp.to_csv(Path(out_path, "BrainSenseLfp.csv"), index=True)
            if plot:
                plotter.brain_sense_lfp_plot(df_brainsense_lfp, self, out_path=out_path)

        if not dfs_bs_td.empty:
            if plot:
                plotter.plot_time_domain_ranges(dfs_bs_td, out_path=out_path)
            for _, df_bs_td_i in tqdm(
                list(enumerate(dfs_bs_td)), desc="BrainSenseTimeDomain Plot Index"
            ):
                str_idx = (
                    df_bs_td_i.index[0].strftime("%Y-%m-%d %H:%M:%S")
                    + f" - {df_bs_td_i.index[-1].strftime('%H:%M:%S')}"
                )
                df_bs_td_i.to_csv(
                    Path(out_path, f"BrainSenseTimeDomain_{str_idx}.csv"),
                    index=True,
                )
                if plot:
                    plotter.plot_df_timeseries(df_bs_td_i, out_path=out_path)
                    plotter.time_frequency_plot_td(
                        df_bs_td_i,
                        indefinite_streaming=False,
                        parser=self,
                        out_path=out_path,
                    )

        if not dfs_is_td.empty:
            if plot:
                plotter.plot_time_domain_ranges(dfs_is_td, out_path=out_path)
            for _, df_is_td_i in tqdm(
                list(enumerate(dfs_is_td)), desc="IndefiniteStreaming Plot Index"
            ):
                str_idx = (
                    df_is_td_i.index[0].strftime("%Y-%m-%d %H:%M:%S")
                    + f" - {df_is_td_i.index[-1].strftime('%H:%M:%S')}"
                )
                df_is_td_i.to_csv(
                    Path(out_path, f"IndefiniteStreaming_{str_idx}.csv"),
                    index=True,
                )
                if plot:
                    plotter.plot_df_timeseries(df_is_td_i, out_path=out_path)
                    plotter.time_frequency_plot_td(
                        df_bs_td_i,
                        indefinite_streaming=True,
                        parser=self,
                        out_path=out_path,
                    )

    def _merge_stim_settings(self, samples) -> pd.DataFrame:
        samples_with_time = samples.reset_index()
        print(samples_with_time)
        # Merge asof backwards will perform a left join, matching each sample with the most
        # recent stim settings that started before sample time.
        samples_with_group = pd.merge_asof(
            samples_with_time.sort_values("Time"),
            self.stim_settings.active_group_history.rename(columns={"hem": "Hemisphere"}),
            left_on="Time",
            right_on="start_time",
            direction="backward",
            by="Hemisphere",
        )

        # Sanity check: if groups don't overlap, all samples should be within start_time and end_time
        if not (samples_with_group["Time"] < samples_with_group["end_time"]).all():
            warnings.warn(
                "Some sample timestamps are outside the group settings time intervals."
                "Check for overlapping groups."
            )

        #  Warn about samples that fell into invalid periods, if thats even possible
        invalid_samples = samples_with_group[~samples_with_group["is_valid"]]
        if not invalid_samples.empty:
            n_invalid = len(invalid_samples)
            total = len(samples_with_group)
            warnings.warn(
                f"{n_invalid}/{total} samples ({100*n_invalid/total:.1f}%) fell into "
                f"INVALID group periods (group switched to undefined configuration). "
                f"These samples dont have defined settings."
            )

        # Drop uninsteresting columns
        return samples_with_group.set_index("Time").drop(
            columns=["start_time", "end_time", "filename"]
        )

    def parse_lfp_trend_logs(
        self,
    ):
        if "LFPTrendLogs" not in self.js["DiagnosticData"]:
            print("No LFPTrendLogs found in the JSON file.")
            return pd.DataFrame()

        # This is out-of-clinic data, recorded between visits

        # There is a group per each 24h, each group indexed by its first datetime
        # Each group has a timepoint for every 10 minutes

        LFPTrendLogs = self.js["DiagnosticData"]["LFPTrendLogs"]
        df = pd.DataFrame(
            [
                {
                    "Time": timepoint["DateTime"],
                    "LFP": timepoint["LFP"],
                    "mA": timepoint["AmplitudeInMilliAmps"],
                    "Hemisphere": hem,
                }
                for hem in ["Left", "Right"]  # For each hem
                # Loop over 24h periods (~days)
                for _, day in LFPTrendLogs[f"HemisphereLocationDef.{hem}"].items()
                for timepoint in day  # Loop over all 10 min-spaced timepoints
            ]
        )

        df["Time"] = pd.to_datetime(df["Time"])  # Vectorized conversion
        df = df.set_index("Time").sort_index()  # Data was likely already sorted
        return df

    def parse_brain_sense_lfp(self):
        if "BrainSenseLfp" not in self.js:
            print("No BrainSenseLfp found in the JSON file.")
            return pd.DataFrame()

        # This is per session data, not out of clinic

        # According to the docs, each time the clinician presses "Start streaming", a new collection
        # of samples is created and they call it "Streaming Sample". I call it "stream" here
        # to avoid confusion with each "sample" within a stream.

        df_idx = []
        for stream in self.js["BrainSenseLfp"]:
            first_packet_time = pd.to_datetime(stream["FirstPacketDateTime"])

            df_stream = pd.DataFrame(
                [
                    {
                        "TicksInMses": sample["TicksInMs"],
                        "Power": sample[hem]["LFP"],
                        "Stim_current": sample[hem]["mA"],
                        "Hemisphere": hem
                    }
                    for hem in ["Left", "Right"]  # For each hem
                    for sample in stream["LfpData"]
                ]
            )

            # Calculate sample times
            df_stream["Time"] = first_packet_time + pd.to_timedelta(
                df_stream["TicksInMses"].diff().fillna(0).cumsum(), unit="ms"
            )

            # Discard TicksInMses
            df_idx.append(df_stream.drop(columns=["TicksInMses"]))

        return (
            pd.concat(df_idx, ignore_index=True).set_index("Time").sort_index()
            # .resample(f"{1 / sfreq:.6f}S")
            # .mean()
        )

    def get_time_stream(
        self, js_td: dict, num_chs: int, verbose: bool = False
    ) -> pd.DataFrame:
        start_time = js_td["FirstPacketDateTime"]
        start_time = pd.Timestamp(start_time)
        fs = js_td["SampleRateInHz"]
        TimeDomainData = np.array(js_td["TimeDomainData"])
        TicksInMses = np.array(
            [int(tick) for tick in js_td["TicksInMses"].split(",")[:-1]]
        )
        TicksDiff = np.diff(TicksInMses)
        GlobalPacketSizes = np.array(
            [int(size) for size in js_td["GlobalPacketSizes"].split(",")[:-1]]
        )
        GlobalSequences = np.array(
            [int(seq) for seq in js_td["GlobalSequences"].split(",")[:-1]]
        )

        if sum(GlobalPacketSizes) != TimeDomainData.shape[0]:
            raise ValueError("GlobalPacketSizes does not match TimeDomainData length")

        df_i = pd.DataFrame(
            {
                "GlobalPacketSizes": GlobalPacketSizes,
                "GlobalSequences": GlobalSequences,
                "TicksInMsesDiff": np.concatenate([np.array([np.nan]), TicksDiff]),
                "TicksInMses": TicksInMses,
            }
        )

        # if indefinite_streaming:
        #     # collapse df_i
        #     idx_250_before = df_i.query("TicksInMsesDiff == 250").index-1

        un, counts = np.unique(TicksDiff, return_counts=True)
        df_counts = pd.DataFrame({"TicksDiff": un, "Counts": counts})

        tdtime = np.arange(0, df_i.iloc[0]["GlobalPacketSizes"] / fs, 1 / fs)
        t_cur = tdtime[-1]

        l_tdtime = []
        l_tdtime.append(tdtime)
        PACKAGE_LOSS_PRESENT = False
        for i in np.arange(1, df_i.shape[0], 1):
            time_diff = df_i.iloc[i]["TicksInMsesDiff"] / 1000  # s

            # for indefinite streaming with 6 chs diffs are 250
            # for brainsense time domain data there are two chs

            # if indefinite_streaming:
            #    time_diff = time_diff / 2  # split in sequence of half
            # if there are two channels, don't adapt time_diff
            if num_chs == 3:
                time_diff = time_diff / 3  # here I don't know,
            elif num_chs == 6:
                time_diff = time_diff / 2
            if time_diff < 0:
                time_diff = np.unique(TicksDiff)[0] / 1000  # s, most freq value

            td_time_packet = np.arange(
                0, df_i.iloc[i]["GlobalPacketSizes"] / fs, 1 / fs
            )

            if time_diff > (df_i.iloc[i]["GlobalPacketSizes"] + 1) / fs:
                PACKAGE_LOSS_PRESENT = True
                td_time_packet += t_cur + time_diff - td_time_packet[-1]
            else:
                td_time_packet += t_cur
                if verbose:
                    print(f"td_time_ shape: {td_time_packet.shape}")
                    print(f"GlobalPacketSizes: {df_i.iloc[i]['GlobalPacketSizes']}\n")
                if td_time_packet.shape[0] != df_i.iloc[i]["GlobalPacketSizes"]:
                    raise ValueError(
                        f"td_time_ shape {td_time_packet.shape} does not match GlobalPacketSizes {df_i.iloc[i]['GlobalPacketSizes']}"
                    )

            t_cur = np.round(td_time_packet[-1] + 1 / fs, decimals=3)
            l_tdtime.append(td_time_packet)
            tdtime = np.concatenate([tdtime, td_time_packet])

        if tdtime.shape[0] != TimeDomainData.shape[0]:
            raise ValueError(
                f"tdtime shape {tdtime.shape} does not match TimeDomainData shape {TimeDomainData.shape}"
            )

        td_ = pd.to_timedelta(tdtime, unit="s") + pd.Timestamp(start_time)
        ch_ = js_td["Channel"]
        df_ch = pd.DataFrame(
            {
                "Time": td_,
                "Data": TimeDomainData,
            }
        )
        df_ch = df_ch.set_index("Time")

        df_ch = df_ch.resample(f"{int(1000 / fs)}ms").mean()
        df_ch["Channel"] = ch_

        if verbose:
            from matplotlib import pyplot as plt

            plt.subplot(1, 2, 1)
            plt.plot(df_ch.query("Channel == @ch_")["Data"].iloc[-500:].values)
            plt.title(f"Corrected Channel {ch_} - TimeDomainData")
            plt.subplot(1, 2, 2)
            plt.plot(TimeDomainData[-500:])
            plt.title(f"JSON TimeDomainData - Channel {ch_}")
            plt.tight_layout()
            plt.xlabel("Samples")

        return df_ch, df_counts, PACKAGE_LOSS_PRESENT

    def read_timedomain_data(self, indefinite_streaming: bool = True) -> pd.DataFrame:
        if indefinite_streaming:
            str_timedomain = "IndefiniteStreaming"
        else:
            str_timedomain = "BrainSenseTimeDomain"
        if str_timedomain not in self.js:
            print(f"No {str_timedomain} found in the JSON file.")
            return []

        FirstPackageDateTimes = np.array(
            [
                self.js[str_timedomain][index_]["FirstPacketDateTime"]
                for index_ in range(len(self.js[str_timedomain]))
            ]
        )
        num_chs = np.where(FirstPackageDateTimes == FirstPackageDateTimes[0])[0].shape[
            0
        ]

        df_ = []
        for package_idx, first_package in tqdm(
            list(enumerate(np.unique(FirstPackageDateTimes))),
            desc=f"{str_timedomain} Index",
        ):
            df_counts_sum = []
            df_chs = []
            idx_package_chs = np.where(FirstPackageDateTimes == first_package)[0]
            for pkg_ch_idx in idx_package_chs:
                df_ch, df_counts, PACKAGE_LOSS_PRESENT = self.get_time_stream(
                    js_td=self.js[str_timedomain][pkg_ch_idx],
                    num_chs=num_chs,
                    verbose=False,
                )
                df_counts["file_idx"] = package_idx
                df_counts_sum.append(df_counts)
                df_chs.append(df_ch)

            df_concat = pd.concat(df_chs, axis=0)
            df_concat = df_concat.reset_index().pivot(
                index="Time", columns="Channel", values="Data"
            )
            df_.append(df_concat)
        return df_

    # def read_brainsense_timedomain(
    #     self,
    # ) -> pd.DataFrame:
    #     if "BrainSenseTimeDomain" not in self.js:
    #         print("No BrainSenseTimeDomain found in the JSON file.")
    #         return pd.DataFrame()

    #     df_counts_sum = []
    #     df_chs = []
    #     for index_ in tqdm(
    #         range(len(self.js["BrainSenseTimeDomain"])), desc="BSTimeDomain Index"
    #     ):
    #         df_ch, df_counts, PACKAGE_LOSS_PRESENT = self.get_time_stream(
    #             self.js["BrainSenseTimeDomain"][index_],
    #             False,
    #             indefinite_streaming=False,
    #         )
    #         df_counts["file_idx"] = index_
    #         df_counts_sum.append(df_counts)

    #         df_chs.append(df_ch)
    #     df_concat = pd.concat(df_chs, axis=0).reset_index()
    #     df_concat = df_concat.drop_duplicates(subset=["Time", "Channel"])
    #     df_concat = df_concat.reset_index().pivot(
    #         index="Time", columns="Channel", values="Data"
    #     )

    #     df_counts = pd.concat(df_counts_sum, axis=0)
    #     df_counts["Time_Diff_Total"] = df_counts["Counts"] * df_counts["TicksDiff"]
    #     file_sum = df_counts.groupby("file_idx")["Time_Diff_Total"].sum().reset_index()
    #     file_missing = (
    #         df_counts.query("TicksDiff != 250")
    #         .groupby("file_idx")["Time_Diff_Total"]
    #         .sum()
    #         .reset_index()
    #     )

    #     df_counts_file_idx = pd.merge(
    #         file_sum,
    #         file_missing,
    #         on="file_idx",
    #         how="outer",
    #         suffixes=("_sum", "_missing"),
    #     )
    #     df_counts_file_idx = df_counts_file_idx.fillna(0)
    #     df_counts_file_idx = df_counts_file_idx.sort_values(
    #         by="Time_Diff_Total_missing", ascending=True
    #     ).iloc[::2]

    #     df_counts_file_idx["Time_Diff_Total_sum"] = (
    #         df_counts_file_idx["Time_Diff_Total_sum"] / 1000
    #     )
    #     df_counts_file_idx["Time_Diff_Total_missing"] = (
    #         df_counts_file_idx["Time_Diff_Total_missing"] / 1000
    #     )

    #     df_counts_file_idx["Time_Diff_Total_missing_clipped"] = df_counts_file_idx[
    #         "Time_Diff_Total_missing"
    #     ].clip(upper=10)

    #     return df_concat, df_counts_file_idx
