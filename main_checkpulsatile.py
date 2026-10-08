"""Run pulsatile artifact rejection and subtraction on a Percept JSON report."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, correlate, find_peaks, sosfiltfilt, welch

from percept_parser.percept import PerceptParser
from percept_parser.pulsatile_rejection import PulsatileArtifactRejector


DEFAULT_REPORT = Path(
    "/mnt/datalake/TRBD-53761/TRBD002/LFP/R/"
    #"Report_Json_Session_Report_20250722T164058.json"
    #"Report_Json_Session_Report_20250805T134908.json"
    #"Report_Json_Session_Report_20251008T170405.json"
    #"Report_Json_Session_Report_20251112T120803.json"
    "Report_Json_Session_Report_20260506T115930.json"
)

#/Volumes/datalake/TRBD-53761/TRBD002/LFP/R/Report_Json_Session_Report_20250805T134908.json


def save_debug_figures(
    report_path: Path, output_directory: Path, lpf_hz: float | None = None
) -> None:
    """Plot per-channel signal and autocorrelation diagnostics before cleaning."""
    parser = PerceptParser(str(report_path), lowpass_hz=lpf_hz)
    rejector = PulsatileArtifactRejector()
    output_directory.mkdir(parents=True, exist_ok=True)

    for stream_name, indefinite_streaming in (
        ("BrainSenseTimeDomain", False),
        ("IndefiniteStreaming", True),
    ):
        streams = parser.js.get(stream_name, [])
        if not streams:
            continue

        sample_rates = {stream["SampleRateInHz"] for stream in streams}
        if len(sample_rates) != 1:
            raise ValueError(
                f"{stream_name} contains multiple sample rates; "
                "cannot safely associate rates with parsed recordings."
            )
        fs = float(sample_rates.pop())
        recordings = parser.read_timedomain_data(
            indefinite_streaming=indefinite_streaming
        )

        for recording_idx, recording in enumerate(recordings):
            recording = rejector.trim_start(recording, fs)
            for channel in recording.columns:
                values = recording[channel].to_numpy(dtype=float)
                finite = np.isfinite(values)
                edges = np.diff(np.r_[False, finite, False].astype(np.int8))
                starts = np.flatnonzero(edges == 1)
                ends = np.flatnonzero(edges == -1)

                for segment_idx, (start, end) in enumerate(zip(starts, ends)):
                    data = values[start:end]
                    if len(data) < round(rejector.minimum_segment_duration_seconds * fs):
                        continue
                    centered = data - np.mean(data)
                    sos = butter(4, [0.5, 30], btype="bandpass", fs=fs, output="sos")
                    filtered = sosfiltfilt(sos, centered)
                    energy = np.dot(filtered, filtered)
                    if not np.isfinite(energy) or energy == 0:
                        continue

                    autocorrelation = correlate(
                        filtered, filtered, mode="full", method="fft"
                    )[len(filtered) - 1 :]
                    autocorrelation /= energy
                    first_lag = max(
                        1,
                        round(rejector.minimum_autocorrelation_lag_seconds * fs),
                    )
                    last_lag = min(
                        round(rejector.maximum_autocorrelation_lag_seconds * fs),
                        len(filtered) // 2,
                    )
                    lag_seconds = np.arange(first_lag, last_lag + 1) / fs
                    autocorrelation_window = autocorrelation[first_lag : last_lag + 1]
                    rms_last_lag = min(
                        round(rejector.autocorrelation_rms_window_seconds * fs),
                        last_lag,
                    )
                    rms_window = autocorrelation[first_lag : rms_last_lag + 1]
                    minimum_interval = round(60 * fs / rejector.maximum_heart_rate_bpm)
                    maximum_interval = round(
                        60 * fs / rejector.minimum_heart_rate_bpm
                    )
                    peaks, _ = find_peaks(
                        autocorrelation_window, distance=minimum_interval
                    )
                    peak_lags = peaks + first_lag
                    intervals = np.diff(peak_lags)
                    physiological = intervals[
                        (intervals >= minimum_interval)
                        & (intervals <= maximum_interval)
                    ]
                    period = (
                        float(np.median(physiological))
                        if len(physiological)
                        else float("nan")
                    )
                    interval_mad = (
                        float(np.median(np.abs(physiological - period)) / fs)
                        if len(physiological)
                        else float("nan")
                    )
                    rms = float(np.sqrt(np.mean(rms_window**2)))
                    windows = rejector._find_pulse_windows(data, fs)
                    enough_intervals = len(physiological) >= 2 and len(
                        physiological
                    ) >= 0.8 * len(intervals)
                    if rms <= rejector.autocorrelation_rms_threshold:
                        gate_status = "RMS gate failed"
                    elif not enough_intervals:
                        gate_status = "too few consistent peak intervals"
                    elif interval_mad > rejector.maximum_interval_mad_seconds:
                        gate_status = "interval-MAD gate failed"
                    elif not windows:
                        gate_status = "template/correlation stage found no windows"
                    else:
                        gate_status = "passed"

                    figure, axes = plt.subplots(
                        3, 1, figsize=(13, 9), constrained_layout=True
                    )
                    preview_end = min(len(data), round(20 * fs))
                    time_seconds = np.arange(preview_end) / fs
                    axes[0].plot(time_seconds, filtered[:preview_end], linewidth=0.7)
                    axes[0].set(
                        title=f"{stream_name} / {channel}: bandpassed signal (first 20 s)",
                        xlabel="Time (s)",
                        ylabel="Amplitude",
                    )
                    for window_start, window_end in windows:
                        if window_start < preview_end and window_end > 0:
                            axes[0].axvspan(
                                max(window_start, 0) / fs,
                                min(window_end, preview_end) / fs,
                                color="tab:red",
                                alpha=0.2,
                            )

                    axes[1].plot(lag_seconds, autocorrelation_window, linewidth=0.8)
                    if len(peaks):
                        axes[1].scatter(
                            peak_lags / fs,
                            autocorrelation_window[peaks],
                            color="tab:red",
                            s=16,
                            label="Autocorrelation peaks",
                        )
                        axes[1].legend()
                    axes[1].axhline(0, color="black", linewidth=0.5)
                    axes[1].set(
                        title=(
                            f"Autocorrelation to {last_lag / fs:.1f} s; "
                            f"RMS (1–{rms_last_lag / fs:.1f} s)={rms:.3f} "
                            f"(threshold {rejector.autocorrelation_rms_threshold:.3f})"
                        ),
                        xlabel="Lag (s)",
                        ylabel="Normalized autocorrelation",
                    )
                    axes[1].text(
                        0.01,
                        0.97,
                        (
                            f"{gate_status}; {len(physiological)} physiological "
                            f"intervals / {len(intervals)} total peak intervals"
                        ),
                        transform=axes[1].transAxes,
                        va="top",
                    )

                    decimation = max(1, len(data) // 50_000)
                    axes[2].plot(
                        np.arange(0, len(data), decimation) / fs,
                        data[::decimation],
                        linewidth=0.5,
                    )
                    for window_start, window_end in windows:
                        axes[2].axvspan(
                            window_start / fs,
                            window_end / fs,
                            color="tab:red",
                            alpha=0.2,
                        )
                    axes[2].set(
                        title=(
                            f"Full raw signal, {len(windows)} selected pulse windows; "
                            f"period={period / fs:.3f} s, "
                            f"interval MAD={interval_mad:.3f} s"
                        ),
                        xlabel="Time (s)",
                        ylabel="Amplitude",
                    )
                    lower, upper = np.nanpercentile(data, [1, 99])
                    if lower < upper:
                        axes[2].set_ylim(lower, upper)

                    filename = (
                        f"{stream_name}_{recording_idx:02d}_"
                        f"{channel}_{segment_idx:02d}.png"
                    )
                    figure.savefig(output_directory / filename, dpi=140)
                    plt.close(figure)
                    print(
                        f"Debug figure: {output_directory / filename} "
                        f"({len(data) / fs:.1f} s, {len(windows)} windows)"
                    )


def finite_runs(values: np.ndarray) -> list[np.ndarray]:
    finite = np.isfinite(values)
    edges = np.diff(np.r_[False, finite, False].astype(np.int8))
    return [
        values[start:end]
        for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
    ]


def welch_spectrum(values: np.ndarray, fs: float, nperseg: int):
    """scipy Welch PSD; NaN gaps are linearly interpolated first since welch needs finite input."""
    finite = np.isfinite(values)
    if finite.sum() < nperseg:
        return None, None
    index = np.arange(len(values))
    filled = np.interp(index, index[finite], values[finite])
    return welch(filled - filled.mean(), fs=fs, nperseg=nperseg)


def save_comparison_figures(
    report_path: Path, output_directory: Path, lpf_hz: float | None = 100
) -> None:
    """Overlay raw and subtracted signals (time domain + PSD) per segment."""
    parser = PerceptParser(str(report_path), lowpass_hz=lpf_hz)
    rejector = PulsatileArtifactRejector()
    output_directory.mkdir(parents=True, exist_ok=True)
    stream_name = "BrainSenseTimeDomain"
    fs = float(parser.js[stream_name][0]["SampleRateInHz"])
    nperseg = int(fs * 4)

    for recording_idx, raw in enumerate(
        parser.read_timedomain_data(indefinite_streaming=False)
    ):
        raw = rejector.trim_start(raw, fs)
        subtracted = rejector.subtract(raw, fs)

        for channel in raw.columns:
            raw_values = raw[channel].to_numpy(dtype=float)
            finite = np.isfinite(raw_values)
            edges = np.diff(np.r_[False, finite, False].astype(np.int8))
            starts = np.flatnonzero(edges == 1)
            ends = np.flatnonzero(edges == -1)

            for segment_idx, (start, end) in enumerate(zip(starts, ends)):
                if end - start < round(rejector.minimum_segment_duration_seconds * fs):
                    continue
                signals = {
                    "raw": raw_values[start:end],
                    "subtracted": subtracted[channel].to_numpy(dtype=float)[start:end],
                }
                changed_fraction = np.mean(
                    ~np.isclose(signals["raw"], signals["subtracted"])
                )
                colors = {
                    "raw": "tab:gray",
                    "subtracted": "tab:blue",
                }

                figure, axes = plt.subplots(
                    3, 1, figsize=(13, 10), constrained_layout=True
                )
                # Skip the first 10 s, which often contains a stimulation/start-up transient.
                zoom_start = min(round(10 * fs), max(0, len(signals["raw"]) - round(10 * fs)))
                zoom = slice(zoom_start, zoom_start + round(10 * fs))
                time_seconds = np.arange(len(signals["raw"]))[zoom] / fs
                for label, values in signals.items():
                    axes[0].plot(
                        time_seconds,
                        values[zoom],
                        label=label,
                        color=colors[label],
                        linewidth=0.7,
                        alpha=0.9,
                    )
                axes[0].legend(loc="upper right")
                axes[0].set(
                    title=(
                        f"{channel} segment {segment_idx} (10 s excerpt, "
                        f"{'LPF ' + format(lpf_hz, 'g') + ' Hz' if lpf_hz else 'no LPF'}); "
                        f"{changed_fraction:.1%} samples changed by subtraction"
                    ),
                    xlabel="Time (s)",
                    ylabel="Amplitude",
                )

                spectra = {}
                for label, values in signals.items():
                    freqs, power = welch_spectrum(values, fs, nperseg)
                    if freqs is None:
                        continue
                    spectra[label] = (freqs, power)
                    axes[1].semilogy(
                        freqs, power, label=label, color=colors[label], linewidth=1
                    )
                    low = freqs <= 10
                    axes[2].plot(
                        freqs[low], 10 * np.log10(power[low]), label=label,
                        color=colors[label], linewidth=1,
                    )
                axes[1].set(
                    title=f"Welch PSD (nperseg={nperseg})",
                    xlabel="Frequency (Hz)",
                    ylabel="Power",
                    xlim=(0, 60),
                )
                axes[1].legend()
                if spectra:
                    in_range = np.concatenate(
                        [p[f <= 60] for f, p in spectra.values()]
                    )
                    axes[1].set_ylim(in_range.min(), in_range.max())
                    low_db = 10 * np.log10(
                        np.concatenate([p[f <= 10] for f, p in spectra.values()])
                    )
                    axes[2].set_ylim(low_db.min(), low_db.max())
                axes[2].set(
                    title="Welch PSD, 0-10 Hz (dB)",
                    xlabel="Frequency (Hz)",
                    ylabel="Power (dB)",
                    xlim=(0, 10),
                )

                filename = (
                    f"compare_{recording_idx:02d}_{channel}_{segment_idx:02d}.png"
                )
                figure.savefig(output_directory / filename, dpi=140)
                plt.close(figure)
                print(
                    f"Comparison figure: {output_directory / filename} "
                    f"(subtracted-changed {changed_fraction:.1%})"
                )


def save_template_figures(
    report_path: Path, output_directory: Path, lpf_hz: float | None = None
) -> None:
    """Plot every subtracted event, the mean template and the scaled template per segment."""
    parser = PerceptParser(str(report_path), lowpass_hz=lpf_hz)
    rejector = PulsatileArtifactRejector()
    output_directory.mkdir(parents=True, exist_ok=True)
    fs = float(parser.js["BrainSenseTimeDomain"][0]["SampleRateInHz"])

    for recording_idx, raw in enumerate(
        parser.read_timedomain_data(indefinite_streaming=False)
    ):
        raw = rejector.trim_start(raw, fs)
        for channel in raw.columns:
            values = raw[channel].to_numpy(dtype=float)
            finite = np.isfinite(values)
            edges = np.diff(np.r_[False, finite, False].astype(np.int8))
            for segment_idx, (start, end) in enumerate(
                zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
            ):
                if end - start < round(rejector.minimum_segment_duration_seconds * fs):
                    continue
                rejector._subtract_segment(values[start:end], fs)
                info = rejector.last_subtraction
                if info is None:
                    continue

                snippets, template = info["snippets"], info["template"]
                centered = snippets - snippets.mean(axis=1, keepdims=True)
                residuals = np.stack(
                    [
                        event - scale * template
                        for event, scale in zip(centered, info["scales"])
                    ]
                )
                time_ms = np.arange(snippets.shape[1]) / fs * 1000
                figure, axes = plt.subplots(
                    1, 3, figsize=(16, 5), constrained_layout=True, sharey=True
                )
                for ax, data, title in (
                    (axes[0], centered, "events (mean removed) + mean template"),
                    (axes[1], info["scales"][:, None] * template, "scaled template per event"),
                    (axes[2], residuals, "residual after subtraction"),
                ):
                    ax.plot(time_ms, data.T, color="tab:gray", alpha=0.25, linewidth=0.6)
                    ax.set(title=title, xlabel="Time (ms)")
                axes[0].plot(time_ms, template, color="tab:red", linewidth=2, label="mean template")
                axes[2].plot(time_ms, residuals.mean(axis=0), color="tab:blue", linewidth=2, label="mean residual")
                axes[0].set_ylabel("Amplitude")
                axes[0].legend()
                axes[2].legend()
                figure.suptitle(
                    f"{channel} rec {recording_idx} seg {segment_idx}: "
                    f"{len(snippets)} events, scale median {np.median(info['scales']):.2f}"
                )
                filename = f"template_{recording_idx:02d}_{channel}_{segment_idx:02d}.png"
                figure.savefig(output_directory / filename, dpi=140)
                plt.close(figure)
                print(f"Template figure: {output_directory / filename}")


def main() -> None:
    argument_parser = argparse.ArgumentParser(
        description=(
            "Parse a Percept report three times: unchanged, with pulsatile artifacts "
            "masked with NaN, and with a fitted pulsatile template subtracted."
        )
    )
    argument_parser.add_argument(
        "report",
        nargs="?",
        type=Path,
        default=DEFAULT_REPORT,
        help=f"Percept JSON report (default: {DEFAULT_REPORT})",
    )
    argument_parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Do not generate parsed-output plots (diagnostic figures are still saved).",
    )
    argument_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for generated CSVs and diagnostic figures.",
    )
    argument_parser.add_argument(
        "--lpf-hz",
        type=float,
        default=100.0,
        help="Low-pass cutoff applied before detection and subtraction (default: 100 Hz).",
    )
    argument_parser.add_argument(
        "--no-lpf",
        action="store_true",
        help="Disable the low-pass filter.",
    )
    args = argument_parser.parse_args()

    report_path: Path = args.report
    output_directory = args.output_dir or Path.cwd() / f"{report_path.stem}_pulsatile_check"
    plot = not args.no_plots

    lpf_hz = None if args.no_lpf else args.lpf_hz

    save_debug_figures(report_path, output_directory / "debug", lpf_hz=lpf_hz)
    save_template_figures(report_path, output_directory / "templates", lpf_hz=lpf_hz)
    save_comparison_figures(report_path, output_directory / "compare", lpf_hz=lpf_hz)

    original_parser = PerceptParser(
        str(report_path),
        verbose=True,
        lowpass_hz=lpf_hz,
    )
    original_parser.parse_all(
        out_path=str(output_directory / "original"),
        plot=plot,
    )

    subtraction_parser = PerceptParser(
        str(report_path),
        verbose=True,
        subtract_pulsatile_artifacts=True,
        lowpass_hz=lpf_hz,
    )
    subtraction_parser.parse_all(
        out_path=str(output_directory / "subtracted"),
        plot=plot,
    )


    rejection_parser = PerceptParser(
        str(report_path),
        verbose=True,
        clean_pulsatile_artifacts=True,
        lowpass_hz=lpf_hz,
    )
    rejection_parser.parse_all(
        out_path=str(output_directory / "rejected"),
        plot=plot,
    )


if __name__ == "__main__":
    main()
