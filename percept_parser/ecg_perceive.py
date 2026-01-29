import numpy as np
from scipy.signal import find_peaks
import pandas as pd
from matplotlib import pyplot as plt

def make_epochs(data: np.ndarray, half_window: int, step: int):
    centers = np.arange(half_window, data.shape[0] - half_window, step)
    offsets = np.arange(-half_window, half_window + 1)
    indices = centers[:, None] + offsets[None, :]
    return data[indices]

def find_highest_peak(data):
    peaks, props = find_peaks(data, width=0)
    if len(peaks) == 0:
        return None, None
    heights = data[peaks]
    max_idx = np.argmax(heights)
    return peaks[max_idx], props["widths"][max_idx]

def temporal_correlation(signal, template):
    corrdata = np.lib.stride_tricks.sliding_window_view(signal, len(template)).T
    corrdata_centered = corrdata - corrdata.mean(axis=0, keepdims=True)
    template_centered = template - template.mean()
    numerator = np.dot(template_centered, corrdata_centered)
    denominator = np.sqrt((template_centered**2).sum()) * np.sqrt((corrdata_centered**2).sum(axis=0))
    r = (numerator / denominator).flatten()**2
    return r

def adaptive_thresholding(r, fs):
    correlation_peaks, _ = find_peaks(r)
    if len(correlation_peaks) == 0:
        return 0.5, 0 # Fallback 

    max_peak_height = np.max(r[correlation_peaks])

    factors = np.linspace(0.95, 0.50, 10)
    thresholds = max_peak_height * factors  # 0.95 to 0.50 in steps 0f 0.05

    max_width_peak = round(0.1 * fs)  # Max peak width = 100ms
    min_physio = None

    threshold_scores = []  # Sum of peaks with 55-120 bpm interval (~heartbeats) / std
    for i, threshold in enumerate(thresholds):
        heartbeat_indices, _ = find_peaks(
            r, height=threshold, width=(None, max_width_peak), distance=min_physio
        )
        if len(heartbeat_indices) < 2:  # Need at least 2 heartbeats to calculate interval
            threshold_scores.append(0)
            continue

        bpm = 60 * fs / np.diff(heartbeat_indices)
        # Avoid division by zero if std is 0 (perfectly regular beat)
        std_bpm = np.std(bpm, ddof=1)
        if std_bpm == 0:
             std_bpm = 1e-6

        score = np.nansum((bpm > 55) & (bpm < 120)) / std_bpm
        threshold_scores.append(0 if np.isnan(score) else score)
        
    best_threshold_idx = np.argmax(threshold_scores)
    return thresholds[best_threshold_idx]

def process_channel_ecg(data, fs):
    """
    Core ECG cleaning logic for a single 1D array.
    Returns: cleaned_data, template, r_correlation, peak_indices
    """
    half_window = round(1 * fs)
    window_step = round(fs)

    epochs = make_epochs(data, half_window, window_step)
    if epochs.shape[0] < 2:
        print("Not enough epochs for processing.")
        return data, np.zeros(1), np.zeros(1), []

    n_epochs = epochs.shape[0]
    len_epoch = epochs.shape[1]

    # Create 2D array to store cross-correlation aligned data
    aligned = np.zeros((n_epochs, 6 * half_window + 1))
    aligned[0, 2 * half_window : 4 * half_window + 1] = epochs[0]

    n = 0
    for i in range(1, epochs.shape[0]):
        current_template = np.mean(aligned[:n+1], axis=0) 
        xcorr = np.correlate(current_template, epochs[i], mode="full")
        max_corr = xcorr.argmax()
        lag = max_corr - len(epochs[i]) - 1
        
        if lag >= 0 and lag + len_epoch <= aligned.shape[1]:
            n += 1
            aligned[n, lag : lag + len_epoch] = epochs[i]
            
    aligned = aligned[:n+1]

    # Template 1
    mean_aligned = np.mean(aligned, 0)
    abs_aligned = np.abs(mean_aligned)

    peaks, _ = find_peaks(abs_aligned)
    if len(peaks) == 0:
        return data, np.zeros(1), np.zeros(1), []

    sorted_indices = np.argsort(abs_aligned[peaks])[::-1][:15]
    max_peak_indices = peaks[sorted_indices]
    
    search_range = 0.05
    found = False
    max_search_range = 0.5 

    left_peak_idx = 0
    right_peak_idx = 0
    left_peak_width = 0
    right_peak_width = 0

    while not found and search_range < max_search_range:
        search_range += 0.025
        start_idx = max_peak_indices[0] - round(fs * search_range)
        end_idx = max_peak_indices[0] + round(fs * search_range)
        
        range_start = max(0, start_idx)
        range_end = min(len(mean_aligned), end_idx)
        
        mean_aligned_range = mean_aligned[range_start:range_end]

        peak_pos, width_pos = find_highest_peak(mean_aligned_range)
        peak_neg, width_neg = find_highest_peak(-mean_aligned_range)

        if peak_pos is not None and peak_neg is not None:
            found = True
            peak_pos_idx = peak_pos + range_start
            peak_neg_idx = peak_neg + range_start

            pos_first = peak_pos_idx < peak_neg_idx
            left_peak_idx = peak_pos_idx if pos_first else peak_neg_idx
            left_peak_width = width_pos if pos_first else width_neg
            right_peak_idx = peak_neg_idx if pos_first else peak_pos_idx
            right_peak_width = width_neg if pos_first else width_pos

    if not found:
        return data, np.zeros(1), np.zeros(1), []

    cut_start = int(left_peak_idx - round(left_peak_width))
    cut_end = int(right_peak_idx + round(right_peak_width))
    cut_start = max(0, cut_start)
    cut_end = min(len(mean_aligned)-1, cut_end)

    ecg_template1 = mean_aligned[cut_start : cut_end + 1]

    # Template 1 Correlation
    r = temporal_correlation(data, ecg_template1)
    
    # Adaptive Thresholding 1
    best_threshold = adaptive_thresholding(r, fs)
    
    min_distance = round(fs / 2)
    peak_indices, _ = find_peaks(r, height=best_threshold, distance=min_distance)
    
    template_peak_idx = np.argmax(np.abs(ecg_template1))
    peak_indices += template_peak_idx         
                               
    pre_window = round(0.05 * fs)
    post_window = round(0.10 * fs)

    new_aligned = np.zeros(pre_window + post_window + 1)
    n = 0
    
    for idx in peak_indices:
        start = idx - pre_window
        end = idx + post_window + 1
        if start >= 0 and end <= len(data):
            n += 1
            new_aligned += data[start:end]
            
    if n == 0:
        return data, np.zeros(1), np.zeros(1), []

    ecg_template2 = new_aligned / n

    # Template 2 Correlation
    r2 = temporal_correlation(data, ecg_template2)
    
    return data, ecg_template2, r2, [] # Return intermediate results for final peak finding outside if simultaneous


def clean_channel_with_indices(data, peak_indices, template_len, plot=False, ch_name=""):
    """
    Applies mirroring cleaning to data given a set of peak indices.
    """
    cleaned_data = data.copy()
    n_peaks = len(peak_indices)
    
    if n_peaks > 0:
        half_len = int(round(template_len / 2))
        
        for idx in peak_indices:
            start_idx = idx
            end_idx = idx + template_len
            
            L_left = half_len
            L_right = template_len - L_left
            
            if (start_idx - L_left + 1 < 0) or (end_idx + L_right > len(data)):
                continue
                
            left_mirror = data[start_idx - L_left + 1 : start_idx + 1][::-1]
            right_mirror = data[end_idx : end_idx + L_right][::-1]
            replacement = np.concatenate((left_mirror, right_mirror))
            
            if len(replacement) == (end_idx - start_idx):
                cleaned_data[start_idx : end_idx] = replacement
            else:
               cleaned_data[start_idx : end_idx] = 0
               
    return cleaned_data


def perceive_ecg(dfs_bs_td : list[pd.DataFrame], fs: int = 250, mode: str = 'independent') -> list[pd.DataFrame]:
    """
    Detects and Removes ECG artifacts from Percept time domain data.
    
    Args:
        dfs_bs_td: List of DataFrames, typically output from `read_timedomain_data`.
        fs: Sampling frequency.
        mode: 'independent' (clean each channel separately) or 'simultaneous' (combine correlations for shared peak detection).
        
    Returns:
        List of DataFrames with cleaned data.
    """
    
    if not dfs_bs_td:
        print("No time domain data provided.")
        return []

    cleaned_dfs = []

    for df_idx, df in enumerate(dfs_bs_td):
        cleaned_df = df.copy()
        
        if mode == 'simultaneous':
            # 1. Calculate correlation signal (r2) for each channel using its own best template
            channel_correlations = []
            channel_templates = {}
            
            valid_channels = []
            
            for col in df.columns:
                data = df[col].to_numpy()
                # Skip if data is flat or invalid
                if np.std(data) == 0: 
                    continue
                    
                # Run detection up to r2 calculation
                _, temp2, r2, _ = process_channel_ecg(data, fs)
                
                # Check if template is valid (not just zeros)
                if len(temp2) > 1 and np.sum(np.abs(temp2)) > 0:
                    channel_correlations.append(r2)
                    channel_templates[col] = temp2
                    valid_channels.append(col)
            
            if not channel_correlations:
                print(f"Session {df_idx}: No valid ECG patterns found in any channel for simult processing.")
                cleaned_dfs.append(cleaned_df)
                continue
                
            # 2. Average the correlation signals
            # Ensure all r2 are same length (should be if data is aligned)
            min_len = min([len(r) for r in channel_correlations])
            avg_r2 = np.mean([r[:min_len] for r in channel_correlations], axis=0)
            
            # 3. Find peaks on the averaged correlation
            best_threshold_avg = adaptive_thresholding(avg_r2, fs)
            max_width_peak = round(0.1 * fs)
            final_peak_indices, _ = find_peaks(avg_r2, height=best_threshold_avg, width=(None, max_width_peak))
            
            print(f"Session {df_idx} (Simultaneous): Found {len(final_peak_indices)} shared beats.")
            
            # 4. Clean each channel using shared peaks
            for col in valid_channels:
                data = df[col].to_numpy()
                template = channel_templates[col]
                cleaned_data = clean_channel_with_indices(data, final_peak_indices, len(template))
                cleaned_df[col] = cleaned_data
                
        else: # Independent mode
            for col in df.columns:
                data = df[col].to_numpy()
                if np.std(data) == 0: continue
                
                # Process fully per channel
                # We need to adapt process_channel_ecg to return r2 so we can find peaks linearly
                # Or just duplicate logic. Let's reuse process_channel_ecg and finish peak finding.
                
                _, temp2, r2, _ = process_channel_ecg(data, fs)
                
                if len(temp2) > 1 and np.sum(np.abs(temp2)) > 0:
                    best_threshold = adaptive_thresholding(r2, fs)
                    max_width_peak = round(0.1 * fs)
                    final_peak_indices, _ = find_peaks(r2, height=best_threshold, width=(None, max_width_peak))
                    
                    print(f"Session {df_idx} Channel {col}: Found {len(final_peak_indices)} beats.")
                    
                    cleaned_data = clean_channel_with_indices(data, final_peak_indices, len(temp2))
                    cleaned_df[col] = cleaned_data
                            
        cleaned_dfs.append(cleaned_df)

    return cleaned_dfs

