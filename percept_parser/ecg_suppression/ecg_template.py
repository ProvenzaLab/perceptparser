import numpy as np
from scipy.signal import find_peaks
from scipy.stats import zscore
from .ecg_remover_base import ECGArtifactRemovalConfig, ECGArtifactRemover
from dataclasses import dataclass
import pandas as pd


@dataclass
class TemplateSubtractionConfig(ECGArtifactRemovalConfig):
    """Configuration for Template Subtraction method."""

    min_height_std: float = 2.5
    min_distance_ms: int = 500  # Minimum distance between peaks in ms
    window_samples_ms: int = 200  # Window size before/after peak in ms
    tail_len: int = 30  # Number of samples for tail equalization
    min_heart_rate_bpm: int = 40


class TemplateSubtractionRemover(ECGArtifactRemover):
    """
    Implements Template Subtraction ECG removal.
    Refactored from ecg_template.py.
    """

    def __init__(self, config: None | TemplateSubtractionConfig = None):
        super().__init__(config or TemplateSubtractionConfig())
        self.config: TemplateSubtractionConfig  # Type hint

    def clean(self, df: pd.DataFrame, fs: float) -> pd.DataFrame:
        cleaned_df = df.copy()
        for col in df.columns:
            signal = df[col].to_numpy()
            if np.std(signal) == 0:
                continue
            cleaned_signal = self._clean_signal(signal, fs)
            cleaned_df[col] = cleaned_signal
        return cleaned_df

    def _clean_signal(self, lfp_signal: np.ndarray, fs: float) -> np.ndarray:
        # Unpack config
        min_height = self.config.min_height_std
        min_distance = int((self.config.min_distance_ms / 1000) * fs)
        win_samples = int((self.config.window_samples_ms / 1000) * fs)

        # --- Step 1: R-peak Detection ---
        lfp_z = zscore(lfp_signal)

        # Detect positive and negative peaks
        peaks_pos, props_pos = find_peaks(
            lfp_z, height=min_height, distance=min_distance
        )
        peaks_neg, props_neg = find_peaks(
            -lfp_z, height=min_height, distance=min_distance
        )

        # Determine orientation
        mean_amp_pos = np.mean(props_pos["peak_heights"]) if len(peaks_pos) > 0 else 0
        mean_amp_neg = np.mean(props_neg["peak_heights"]) if len(peaks_neg) > 0 else 0

        if mean_amp_neg > mean_amp_pos and len(peaks_neg) >= len(peaks_pos) * 0.8:
            r_peaks = peaks_neg
        else:
            r_peaks = peaks_pos

        # Safety check for heart rate
        if len(r_peaks) == 0:
            return lfp_signal

        # --- Step 2: Template Generation ---
        epochs = []
        valid_peaks = []

        for r in r_peaks:
            start = r - win_samples
            end = r + win_samples
            if start >= 0 and end < len(lfp_signal):
                epochs.append(lfp_signal[start:end])
                valid_peaks.append(r)

        if not epochs:
            return lfp_signal

        epochs_arr = np.array(epochs)
        raw_template = np.mean(epochs_arr, axis=0)

        # Equalizing Tails
        tail_len = self.config.tail_len
        if len(raw_template) < 2 * tail_len:
            tail_len = len(raw_template) // 2

        t_start_seg = raw_template[:tail_len]
        t_end_seg = raw_template[-tail_len:]

        min_diff = np.inf
        cut_start = 0
        cut_end = 0

        for i, val_start in enumerate(t_start_seg):
            for j, val_end in enumerate(t_end_seg):
                diff = abs(val_start - val_end)
                if diff < min_diff:
                    min_diff = diff
                    cut_start = i
                    cut_end = (len(raw_template) - tail_len) + j

        final_template = raw_template[cut_start : cut_end + 1].copy()

        # Fix offset (set ends to same value)
        target_val = max(final_template[0], final_template[-1])
        final_template[0] = target_val
        final_template[-1] = target_val

        # Optimization and Subtraction
        lfp_cleaned = lfp_signal.copy()
        template_rel_start = -win_samples + cut_start

        for i, r in enumerate(valid_peaks):
            seg_start = r + template_rel_start
            seg_end = seg_start + len(final_template)

            if seg_start < 0 or seg_end > len(lfp_signal):
                continue

            lfp_epoch = lfp_signal[seg_start:seg_end]

            # Linear regression to find scale and offset (Optimization)
            if len(lfp_epoch) == len(final_template):
                scale, offset = np.polyfit(final_template, lfp_epoch, 1)
                artifact_model = (scale * final_template) + offset

                # Subtraction
                lfp_cleaned[seg_start:seg_end] = lfp_epoch - artifact_model

        return lfp_cleaned
