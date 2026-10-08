import numpy as np
import pandas as pd
from scipy.signal import butter, correlate, find_peaks, sosfiltfilt


def lowpass_dataframe(df: pd.DataFrame, fs: float, cutoff_hz: float) -> pd.DataFrame:
    """Zero-phase Butterworth low-pass applied to each contiguous finite run per column."""
    if cutoff_hz >= fs / 2:
        raise ValueError("Low-pass cutoff must be below the Nyquist frequency")
    sos = butter(4, cutoff_hz, btype="lowpass", fs=fs, output="sos")
    filtered_df = df.copy()
    for column_idx in range(df.shape[1]):
        data = df.iloc[:, column_idx].to_numpy(dtype=float)
        filtered = data.copy()
        finite = np.isfinite(data)
        edges = np.diff(np.r_[False, finite, False].astype(np.int8))
        for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            # sosfiltfilt needs more samples than its default padding length
            if end - start > 3 * (2 * len(sos) + 1):
                filtered[start:end] = sosfiltfilt(sos, data[start:end])
        filtered_df.iloc[:, column_idx] = filtered
    return filtered_df


class PulsatileArtifactRejector:
    """Detect and remove repetitive, physiologically paced pulsatile artifacts."""

    autocorrelation_rms_threshold = 0.081
    autocorrelation_rms_window_seconds = 10
    minimum_heart_rate_bpm = 50
    maximum_heart_rate_bpm = 120
    maximum_autocorrelation_lag_seconds = 30
    minimum_autocorrelation_lag_seconds = 1
    maximum_interval_mad_seconds = 0.1
    minimum_segment_duration_seconds = 20
    initial_trim_seconds = 10
    # Subtraction template spans one full cycle around each beat.
    template_pre_period_fraction = 0.5
    template_post_period_fraction = 0.5

    def clean(self, df: pd.DataFrame, fs: float) -> pd.DataFrame:
        """Mask detected pulse intervals with NaN, preserving the time index."""
        return self._process_dataframe(df, fs, subtract=False)

    def subtract(self, df: pd.DataFrame, fs: float) -> pd.DataFrame:
        """Subtract a fitted pulsatile template from detected pulse intervals."""
        return self._process_dataframe(df, fs, subtract=True)

    def trim_start(self, df: pd.DataFrame, fs: float) -> pd.DataFrame:
        """Set the first initial_trim_seconds of the dataframe to NaN (start-up transient)."""
        trimmed = df.copy()
        trimmed.iloc[: round(self.initial_trim_seconds * fs)] = np.nan
        return trimmed

    def _process_dataframe(
        self, df: pd.DataFrame, fs: float, subtract: bool
    ) -> pd.DataFrame:
        if fs <= 60:
            raise ValueError("Pulsatile artifact rejection requires a sample rate above 60 Hz")

        cleaned_df = self.trim_start(df, fs)
        df = cleaned_df.copy()
        for column_idx in range(df.shape[1]):
            data = df.iloc[:, column_idx].to_numpy(dtype=float)
            finite = np.isfinite(data)
            segment_edges = np.diff(np.r_[False, finite, False].astype(np.int8))
            starts = np.flatnonzero(segment_edges == 1)
            ends = np.flatnonzero(segment_edges == -1)

            for start, end in zip(starts, ends):
                if end - start < round(self.minimum_segment_duration_seconds * fs):
                    continue
                segment = data[start:end]
                if subtract:
                    result = self._subtract_segment(segment, fs)
                else:
                    result = self._mask_segment(segment, fs)
                cleaned_df.iloc[start:end, column_idx] = result
        return cleaned_df

    def _detect_period(self, data: np.ndarray, fs: float):
        """Return (bandpassed signal, period in samples, minimum interval) or None."""
        centered = data - np.mean(data)
        if np.std(centered) == 0:
            return None

        sos = butter(4, [0.5, 30], btype="bandpass", fs=fs, output="sos")
        filtered = sosfiltfilt(sos, centered)
        energy = np.dot(filtered, filtered)
        if not np.isfinite(energy) or energy == 0:
            return None

        autocorrelation = correlate(filtered, filtered, mode="full", method="fft")[
            len(filtered) - 1 :
        ]
        autocorrelation /= energy

        first_lag = max(1, round(self.minimum_autocorrelation_lag_seconds * fs))
        last_lag = min(
            round(self.maximum_autocorrelation_lag_seconds * fs),
            len(filtered) // 2,
        )
        if last_lag <= first_lag:
            return None

        autocorrelation_window = autocorrelation[first_lag : last_lag + 1]
        rms_last_lag = min(
            round(self.autocorrelation_rms_window_seconds * fs), last_lag
        )
        rms_window = autocorrelation[first_lag : rms_last_lag + 1]
        if (
            np.sqrt(np.mean(np.square(rms_window)))
            <= self.autocorrelation_rms_threshold
        ):
            return None

        minimum_interval = round(60 * fs / self.maximum_heart_rate_bpm)
        maximum_interval = round(60 * fs / self.minimum_heart_rate_bpm)
        autocorrelation_peaks, _ = find_peaks(
            autocorrelation_window, distance=minimum_interval
        )
        peak_lags = autocorrelation_peaks + first_lag
        intervals = np.diff(peak_lags)
        physiological_intervals = intervals[
            (intervals >= minimum_interval) & (intervals <= maximum_interval)
        ]
        if len(physiological_intervals) < 2 or len(physiological_intervals) < 0.8 * len(
            intervals
        ):
            return None

        period = float(np.median(physiological_intervals))
        interval_mad = np.median(np.abs(physiological_intervals - period)) / fs
        if interval_mad > self.maximum_interval_mad_seconds:
            return None

        return filtered, period, minimum_interval

    def _find_pulse_windows(
        self, data: np.ndarray, fs: float
    ) -> list[tuple[int, int]]:
        detected = self._detect_period(data, fs)
        if detected is None:
            return []
        filtered, period, minimum_interval = detected
        cycle_length = round(period)
        cycle_count = len(filtered) // cycle_length
        if cycle_count < 2:
            return []
        cycle_template = filtered[: cycle_count * cycle_length].reshape(
            cycle_count, cycle_length
        ).mean(axis=0)
        pulse_peak = int(np.argmax(np.abs(cycle_template)))
        cycle_template = np.roll(
            cycle_template, cycle_length // 2 - pulse_peak
        )
        pulse_peak = int(np.argmax(np.abs(cycle_template)))

        zero_crossings = np.flatnonzero(
            cycle_template[:-1] * cycle_template[1:] <= 0
        )
        left_crossings = zero_crossings[zero_crossings < pulse_peak]
        right_crossings = zero_crossings[zero_crossings >= pulse_peak]
        if len(left_crossings) == 0 or len(right_crossings) == 0:
            return []

        template_start = int(left_crossings[-1] + 1)
        template_end = int(right_crossings[0] + 1)
        pulse_template = cycle_template[template_start:template_end].copy()
        if len(pulse_template) < 3:
            return []
        pulse_template -= np.mean(pulse_template)

        correlations = np.abs(
            correlate(filtered, pulse_template, mode="valid", method="fft")
        )
        maximum_correlation = np.max(correlations)
        if not np.isfinite(maximum_correlation) or maximum_correlation == 0:
            return []

        pulse_starts, _ = find_peaks(
            correlations,
            height=0.1 * maximum_correlation,
            distance=minimum_interval,
        )
        if len(pulse_starts) == 0:
            return []

        pulse_centers = pulse_starts + pulse_peak - template_start
        windows = []
        for center in pulse_centers:
            start = max(0, center - pulse_peak + template_start)
            end = min(len(data), center + template_end - pulse_peak)
            if end > start:
                windows.append((start, end))
        return windows

    def _mask_segment(self, data: np.ndarray, fs: float) -> np.ndarray:
        rejected = data.copy()
        for start, end in self._find_pulse_windows(data, fs):
            rejected[start:end] = np.nan
        return rejected

    def _find_beats(self, filtered: np.ndarray, period: float):
        """Locate beats by iteratively refining a beat-aligned template."""
        pre = int(round(self.template_pre_period_fraction * period))
        post = int(round(self.template_post_period_fraction * period))
        distance = max(1, int(0.7 * period))

        def aligned_template(beats):
            valid = [b for b in beats if b - pre >= 0 and b + post <= len(filtered)]
            if len(valid) < 2:
                return None
            return np.mean([filtered[b - pre : b + post] for b in valid], axis=0)

        best = None
        for sign in (1.0, -1.0):
            peaks, _ = find_peaks(sign * filtered, distance=distance)
            if len(peaks) == 0:
                continue
            strength = np.mean(sign * filtered[peaks])
            if best is None or strength > best[0]:
                best = (strength, peaks)
        if best is None:
            return None
        beats = best[1]

        for _ in range(3):
            template = aligned_template(beats)
            if template is None or np.dot(template, template) == 0:
                return None
            template = template - template.mean()
            correlations = correlate(filtered, template, mode="valid", method="fft")
            # correlations[i] aligns the template start with sample i
            peaks, _ = find_peaks(correlations, distance=distance)
            if len(peaks) == 0:
                return None
            keep = correlations[peaks] > 0.3 * np.max(correlations[peaks])
            beats = peaks[keep] + pre
        return beats, pre, post

    def _subtract_segment(self, data: np.ndarray, fs: float) -> np.ndarray:
        self.last_subtraction = None
        detected = self._detect_period(data, fs)
        if detected is None:
            return data
        filtered, period, _ = detected
        found = self._find_beats(filtered, period)
        if found is None:
            return data
        beats, pre, post = found
        full_windows = [
            (b - pre, b + post)
            for b in beats
            if b - pre >= 0 and b + post <= len(data)
        ]
        if len(full_windows) < 2:
            return data

        snippets = np.stack([data[start:end] for start, end in full_windows])
        template = np.mean(snippets, axis=0)
        template -= np.mean(template)
        template_energy = np.dot(template, template)
        if not np.isfinite(template_energy) or template_energy <= np.finfo(float).eps:
            return data

        cleaned = data.copy()
        scales = []
        for start, end in full_windows:
            event = data[start:end]
            scale = np.dot(event - np.mean(event), template) / template_energy
            scales.append(scale)
            # Fit on the original data and accumulate, so overlapping windows are not subtracted from twice-modified data.
            cleaned[start:end] -= scale * template
        # Kept for diagnostics: the events, template and per-event scales used.
        self.last_subtraction = {
            "windows": full_windows,
            "snippets": snippets,
            "template": template,
            "scales": np.array(scales),
        }
        return cleaned
