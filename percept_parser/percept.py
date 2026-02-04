from datetime import datetime
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
from typing import Optional

from pathlib import Path
from .stim_settings import FileStimGroupSettings
from .ecg_suppression import (
    TemplateSubtractionRemover,
    PerceiveToolboxRemover,
    TemplateSubtractionConfig,
    PerceiveToolboxConfig,
)


class PerceptParser:
    def __init__(
        self,
        filename: str,
        verbose: bool = False,
        ecg_method: Optional[str] = None,
        ecg_params: Optional[dict] = None,
    ):
        """
        Parses Percept JSON reports into pandas DataFrames.

        Parameters
        ----------
        filename : str
            Path to the JSON report file.
        verbose : bool, optional
            If True, prints progress and info messages. Default is False.
        ecg_method : str, optional
            Method to use for ECG artifact removal. Options: "template", "perceive".
            Default is None (no removal).
        ecg_params : dict, optional
            Dictionary of parameters to pass to the ECG removal configuration.
            See `TemplateSubtractionConfig` and `PerceiveToolboxConfig` for available options.
        """
        self.filename = filename
        self.verbose = verbose

        self.ecg_remover = None
        if ecg_method:
            if ecg_method.lower() == "template":
                config = TemplateSubtractionConfig(**(ecg_params or {}))
                self.ecg_remover = TemplateSubtractionRemover(config)
            elif ecg_method.lower() == "perceive":
                config = PerceiveToolboxConfig(**(ecg_params or {}))
                self.ecg_remover = PerceiveToolboxRemover(config)
            else:
                warnings.warn(f"Unknown ECG method: {ecg_method}")

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

        try:
            self.stim_settings = FileStimGroupSettings(self.js, self.filename)
        except Exception as e:
            print(f"Error initializing stim settings: {e}")
            self.stim_settings = None
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
            if self.stim_settings is not None:
                df_lfp_trend_logs = self._merge_stim_settings(df_lfp_trend_logs)
            df_lfp_trend_logs.to_csv(Path(out_path, "LFPTrendLogs.csv"), index=True)
            if plot:
                plotter.lfptrendlog_plot(df_lfp_trend_logs, self, path_out=out_path)

        if not df_brainsense_lfp.empty:
            if self.stim_settings is not None:
                df_brainsense_lfp = self._merge_stim_settings(df_brainsense_lfp)
            df_brainsense_lfp.to_csv(Path(out_path, "BrainSenseLfp.csv"), index=True)
            if plot:
                df_left = df_brainsense_lfp.query("Hemisphere == 'Left'")
                df_right = df_brainsense_lfp.query("Hemisphere == 'Right'")
                df_left.columns = [f"Left_{col}" for col in df_left.columns]
                df_right.columns = [f"Right_{col}" for col in df_right.columns]
                df_brainsense_lfp_comb = pd.concat([df_left, df_right], axis=1)

                plotter.brain_sense_lfp_plot(
                    df_brainsense_lfp_comb, self, out_path=out_path
                )

        if len(dfs_bs_td) > 0:
            if plot:
                plotter.plot_time_domain_ranges(dfs_bs_td, out_path=out_path)
            for _, df_bs_td_i in tqdm(
                list(enumerate(dfs_bs_td)), desc="BrainSenseTimeDomain Plot Index"
            ):
                str_idx = (
                    df_bs_td_i.index[0].strftime("%Y-%m-%d_%H-%M-%S")
                    + f"_{df_bs_td_i.index[-1].strftime('%H-%M-%S')}"
                )
                # pivot each column to a channel
                df_bs_td_i_pivot = df_bs_td_i.reset_index().melt(
                    id_vars=["Time"], var_name="Channel", value_name="Value"
                )
                df_bs_td_i_pivot["Hemisphere"] = np.where(
                    df_bs_td_i_pivot["Channel"].str.contains("LEFT"), "Left", "Right"
                )
                if self.stim_settings is not None:
                    df_bs_td_i_pivot = self._merge_stim_settings(df_bs_td_i_pivot)

                df_bs_td_i_pivot.to_csv(
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

        if len(dfs_is_td) > 0:
            if plot:
                plotter.plot_time_domain_ranges(dfs_is_td, out_path=out_path)
            for _, df_is_td_i in tqdm(
                list(enumerate(dfs_is_td)), desc="IndefiniteStreaming Plot Index"
            ):
                str_idx = (
                    df_is_td_i.index[0].strftime("%Y-%m-%d_%H-%M-%S")
                    + f"_{df_is_td_i.index[-1].strftime('%H-%M-%S')}"
                )
                df_is_td_i_pivot = df_is_td_i.reset_index().melt(
                    id_vars=["Time"], var_name="Channel", value_name="Value"
                )
                df_is_td_i_pivot["Hemisphere"] = np.where(
                    df_is_td_i_pivot["Channel"].str.contains("LEFT"), "Left", "Right"
                )
                if self.stim_settings is not None:
                    df_is_td_i_pivot = self._merge_stim_settings(df_is_td_i_pivot)
                else:
                    # set Time as index
                    df_is_td_i_pivot = df_is_td_i_pivot.set_index("Time")
                df_is_td_i_pivot.to_csv(
                    Path(out_path, f"IndefiniteStreaming_{str_idx}.csv"),
                    index=True,
                )
                if plot:
                    plotter.plot_df_timeseries(df_is_td_i, out_path=out_path)
                    plotter.time_frequency_plot_td(
                        df_is_td_i,
                        indefinite_streaming=True,
                        parser=self,
                        out_path=out_path,
                    )

    def _merge_stim_settings(self, samples) -> pd.DataFrame:
        samples_with_time = samples.reset_index()

        # Make sure time has same format in both tables before merging
        if samples_with_time["Time"].dtype != "datetime64[ns, UTC]":
            samples_with_time["Time"] = samples_with_time["Time"].astype(
                "datetime64[ns, UTC]"
            )

        stim_history = self.stim_settings.active_group_history.rename(
            columns={"hem": "Hemisphere"}
        )
        if stim_history["start_time"].dtype != "datetime64[ns, UTC]":
            stim_history["start_time"] = stim_history["start_time"].astype(
                "datetime64[ns, UTC]"
            )

        # Merge asof backwards will perform a left join, matching each sample with the most
        # recent stim settings that started before sample time.
        samples_with_group = pd.merge_asof(
            samples_with_time.sort_values("Time"),
            stim_history,
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
                f"{n_invalid}/{total} samples ({100 * n_invalid / total:.1f}%) fell into "
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
                        "Hemisphere": hem,
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
    ) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
        """Estimate relative time at which each datapoint in a BrainSenseTimeDomain
        or IndefiniteStreaming data stream was sampled.  Uses the algorithm defined
        in page 35 of Medtronic's DBS Sensing And Adaptive Therapy White Paper

        Args:
            stream_json (dict): _description_
            num_chs (int): _description_
            verbose (bool, optional): _description_. Defaults to False.

        Raises:
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, bool]: (df_ch, df_counts, PACKAGE_LOSS_PRESENT)
        """
        # Absolute datetime for the first sent pacakge of the stream
        first_packet_time = datetime.fromisoformat(js_td["FirstPacketDateTime"])

        # Recording sample rate for this data stream
        fs = js_td["SampleRateInHz"]

        # Measurement value for each datapoint at SampleRateInHz
        TimeDomainData = np.array(js_td["TimeDomainData"])

        #  List of number of samples per telemetry packet
        GlobalPacketSizes = np.fromstring(
            js_td["GlobalPacketSizes"], sep=",", dtype=int
        )

        # Recording device system tick at the moment of package transmission
        TicksInMses = np.fromstring(js_td["TicksInMses"], sep=",", dtype=int)

        # Integrity check: The sum of packet sizes should match the namber of data points
        if np.sum(GlobalPacketSizes) != TimeDomainData.shape[0]:
            raise ValueError("GlobalPacketSizes does not match TimeDomainData length")

        # Integrity check: The amount of packets should match the array of packet send times
        if len(GlobalPacketSizes) != len(TicksInMses):
            raise ValueError("Inconsistent number of packets in stream")

        # Differences in tick between sent packages, to check for package loss
        TicksDiff = np.concatenate([np.diff(TicksInMses), np.array([np.nan])])

        # Check that packages are sorteds (sometimes this fails, which is worrysome)
        # if not np.all(TicksInMses[:-1] <= TicksInMses[1:]):
        #     raise ValueError("Packages in the wrong order")

        # if indefinite_streaming:
        #     # collapse df_i
        #     idx_250_before = df_i.query("TicksInMsesDiff == 250").index-1

        un, counts = np.unique(np.diff(TicksInMses), return_counts=True)
        df_counts = pd.DataFrame({"TicksDiff": un, "Counts": counts})

        # According to the algorithm described in the whitepaper, we have to iterate backwards
        period_ms = 1000 / fs  # Gap between data points in milliseconds
        num_packets = len(GlobalPacketSizes)

        # Start with he last packet
        last_packet_size = GlobalPacketSizes[-1]
        tick_ms_end = TicksInMses[-1]
        start_time = tick_ms_end - (last_packet_size - 1) * period_ms
        PACKAGE_LOSS_PRESENT = False
        tdtime = np.linspace(start_time, tick_ms_end, last_packet_size)

        # Continue from second to last and backwards
        for i in range(num_packets - 2, -1, -1):
            time_diff = TicksDiff[i] / 1000  # s

            # for indefinite streaming with 6 chs diffs are 250
            # for brainsense time domain data there are two chs

            # if indefinite_streaming:
            #    time_diff = time_diff / 2  # split in sequence of half
            # if there are two channels, don't adapt time_diff
            if num_chs == 3:
                time_diff = time_diff / 3  # here I don't know,
            elif num_chs == 6:
                time_diff = time_diff / 2

            # Is this even possible? If so, would need to reorder TimeDomainData
            # if time_diff < 0:
            #     time_diff = np.bincount(TicksDiff).argmax() / 1000  # most freq value

            # Should I use this one or previous one for time diff check?
            packet_size = GlobalPacketSizes[i]

            # In case of packet loss (time between packets too long)
            if time_diff > (packet_size + 1) / fs:  # If packages missing
                PACKAGE_LOSS_PRESENT = True
                end_time = TicksInMses[i]  # Re-anchor time using the ticks
                start_time = end_time - (packet_size - 1) * period_ms
                tdtime_packet = np.linspace(start_time, end_time, packet_size)
                # And join with the previous
                tdtime = np.concat((tdtime_packet, tdtime))
            else:
                # If no packet loss, assume data is continuous
                end_time = tdtime[0] - period_ms  # Account for gap betwen packets
                start_time = end_time - (packet_size - 1) * period_ms
                tdtime_packet = np.linspace(start_time, end_time, packet_size)

                if verbose:
                    print(f"td_time_ shape: {tdtime_packet.shape}")
                    print(f"GlobalPacketSizes: {GlobalPacketSizes[i]}\n")
                if tdtime_packet.shape[0] != GlobalPacketSizes[i]:
                    raise ValueError(
                        f"td_time_ shape {tdtime_packet.shape} does not match GlobalPacketSizes {GlobalPacketSizes[i]}"
                    )
                tdtime = np.concat((tdtime_packet, tdtime))

        tdtime -= tdtime[0]  # Start at 0
        tdtime /= 1000  # To seconds

        # Assert that the time array has the same shape as the measurements array
        # (i.e. all measurements have been assigned a timestamp)
        if tdtime.shape[0] != TimeDomainData.shape[0]:
            raise ValueError(
                f"tdtime shape {tdtime_packet.shape} does not match TimeDomainData shape {TimeDomainData.shape}"
            )

        # Convert to dataframe
        td_ = pd.to_timedelta(tdtime, unit="s") + pd.Timestamp(first_packet_time)
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

    def read_timedomain_data(
        self, indefinite_streaming: bool = True
    ) -> list[pd.DataFrame]:
        str_timedomain = (
            "IndefiniteStreaming" if indefinite_streaming else "BrainSenseTimeDomain"
        )
        if str_timedomain not in self.js:
            print(f"No {str_timedomain} found in the JSON file.")
            return []

        # Each channel has its own "Streaming Sample" object,
        # but all recording sessions are stored at the same level
        td_data = self.js[str_timedomain]

        # We can separate recording sessions looking at FirstPacketDateTime
        FirstPackageDateTimes = np.array(
            [stream["FirstPacketDateTime"] for stream in td_data]
        )
        channels = np.unique([stream["Channel"] for stream in td_data])
        num_chs = len(channels)

        df_ = []
        for package_idx, first_package in tqdm(
            list(enumerate(np.unique(FirstPackageDateTimes))),
            desc=f"{str_timedomain} Index",
        ):
            df_counts_sum = []
            df_chs = []
            idx_package_chs = np.where(FirstPackageDateTimes == first_package)[0]
            for pkg_ch_idx in idx_package_chs:
                try:
                    df_ch, df_counts, PACKAGE_LOSS_PRESENT = self.get_time_stream(
                        js_td=self.js[str_timedomain][pkg_ch_idx],
                        num_chs=num_chs,
                        verbose=False,
                    )
                except Exception as e:
                    print(e)

                df_counts["file_idx"] = package_idx
                df_counts_sum.append(df_counts)
                df_chs.append(df_ch)

            df_concat = pd.concat(df_chs, axis=0)
            df_concat = df_concat.reset_index().pivot(
                index="Time", columns="Channel", values="Data"
            )

            if self.ecg_remover is not None:
                try:
                    # Get fs from the first package in this stream/session
                    fs = 250  # Default
                    if len(idx_package_chs) > 0:
                        fs = self.js[str_timedomain][idx_package_chs[0]][
                            "SampleRateInHz"
                        ]

                    if self.verbose:
                        print(
                            f"Applying ECG cleaning ({type(self.ecg_remover).__name__}) to stream {package_idx}, fs={fs}"
                        )

                    df_concat = self.ecg_remover.clean(df_concat, fs)
                except Exception as e:
                    print(f"Error applying ECG removal: {e}")

            df_.append(df_concat)
        return df_
