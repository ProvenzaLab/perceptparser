import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from datetime import timedelta
from datetime import time as dttime
from zoneinfo import ZoneInfo
from scipy.ndimage import gaussian_filter1d

central_time = ZoneInfo('America/Chicago')

def zscore_group(group, cols_to_zscore=['lfp_left_raw', 'lfp_right_raw']):
    """
    Calculate Z-scored version of specified data.

    Parameters:
    - group (pd.DataFrame): group dataframe to Z-score across.
    - cols_to_zscore (list): columns to calculate and return Z-scored version of.

    Returns:
    - cols_z_scored_df (pd.DataFrame): DataFrame containing only Z-scored version of provided columns. Output is NaN wherever input was NaN.
    """
    new_cols = {}
    for col in cols_to_zscore:
        # Write new column name for easy merging later.
        if '_raw' in col:
            zscored_col_name = col.replace('_raw', '_z_scored')
        elif '_filled' in col:
            zscored_col_name = col.replace('_filled', '_z_scored')
        else:
            zscored_col_name = col + '_z_scored'

        if pd.notna(group[col]).sum() > 1:
            # Z-score data: for each data point, subtract mean and divide by standard deviation of entire series.
            mean = np.nanmean(group[col], axis=0)
            std = np.nanstd(group[col], axis=0)
            new_cols[zscored_col_name] = (group[col] - mean) / std if std != 0 else [np.nan] * len(group[col])
        else:
            # If all values are nan, return np.nan
            new_cols[zscored_col_name] = [np.nan] * len(group)
    
    # Create new dataframe containing only the new Z-scored columns and return it. Keep the same indices as the input group for easy merging.
    return pd.DataFrame(new_cols, index=group.index)

def correct_timestamps(group, timestamp_col='timestamp', threshold=pd.Timedelta(minutes=10, seconds=1)):
    """
    Adjust timestamps in each group where intervals are off by one second.
    
    Parameters:
    - group: DataFrame containing a column with timestamps.
    - timestamp_col: Name of the dataframe column containing the timestamp.
    - threshold: Maximum expected time difference for a 10-minute interval (default: 10:01).
    
    Returns:
    - Timestamp column with corrected timestamps.
    """
    timestamps = group[timestamp_col].sort_values()
    time_diff = timestamps.diff()

    correction_mask = time_diff == threshold
    reset_mask = time_diff > threshold
    correction_groups = reset_mask.cumsum()
    correction = correction_mask.groupby(correction_groups).cumsum() * -pd.Timedelta(seconds=1)

    return pd.DataFrame(timestamps + correction)

def get_contig(group, col_to_check, contig_colname, time_bin_col='time_bin'):
    """
    Label each row in a contiguous series with the same number, and return a DataFrame containing those labels.
    Contiguous data is a series of data where each data point is separated by ~10 minutes (with margin of error 1 second either direction).

    Parameters:
    - group (pd.DataFrame): DataFrame to get contiguous data labels of.
    - col_to_check (str): Name of the column you are interested in getting contiguous data labels for. Rows containing NaN values in this column will be dropped prior to generating labels.
    - contig_colname (str): Name of the column in the returned DataFrame containing contiguous data labels.
    - time_bin_col (str): Name of the column containing data time bin timestamps.

    Returns:
    - contig_df (pd.DataFrame): DataFrame containing only labels for sections of contiguous data, where indices match indices of the input.
    """
    # Drop any NaN values of the column to get contiguous labels for.
    group = group.dropna(subset=col_to_check)

    # Ensure values are sorted by timestamp.
    group = group.sort_values(time_bin_col)

    time_diff = group[time_bin_col].diff()
    contig = (time_diff != timedelta(minutes=10)).cumsum()-1

    return pd.DataFrame({contig_colname: contig}, index=group.index, dtype='Int64')

def add_empty_rows(group, pt_id, lead_location, time_bin_col='time_bin', dbs_on_date=None):
    """
    Fills in dataframe holes so that all possible empty time bins contain NaN. Useful for interpolation later.

    Parameters:
    - group (pd.DataFrame): DataFrame containing a single patient's data from a single lead.
    - lead_location (str): Lead location for this lead.
    - time_bin_col (str, optional): Name of the DataFrame column containing the datas' time bins timestamps.
    - dbs_on_date (datetime, optional): Date when the DBS was turned on. If provided, will fill the "days_since_dbs" column.

    Returns:
    - interp_df (pd.DataFrame): DataFrame containing empty rows where no data was recorded by the Percept device.
    """

    # Get sizes of gaps of missing data in terms of number of missing data points.
    gap_sizes = (np.diff(group[time_bin_col]) // timedelta(minutes=10)).astype(int)
    small_gap_start_inds = np.where(gap_sizes >= 2)[0]
    gap_sizes = gap_sizes[small_gap_start_inds]
    
    # Get the last time bin timestamp before each unfilled data gap so we know where to start filling from.
    gap_start_times = group.loc[group.index[small_gap_start_inds], time_bin_col]
    times_to_fill = [gap_start_time + timedelta(minutes=10) * i for (gap_start_time, gap_size) in zip(gap_start_times, gap_sizes) for i in range(1, gap_size)]
    
    # Create new dataframe and fill in information in relevant columns.
    if len(times_to_fill) != 0:
        interp_df = pd.DataFrame()
        interp_df['timestamp'] = times_to_fill # Timestamp is set to time bin
        interp_df[time_bin_col] = times_to_fill # Time bin is where the data was not recorded/missing from the device.
        interp_df['CT_timestamp'] = interp_df['timestamp'].dt.tz_convert(central_time)
        if dbs_on_date is not None:
            interp_df['days_since_dbs'] = [dt.days for dt in (interp_df['CT_timestamp'].dt.date - dbs_on_date)]
        interp_df['lead_location'] = lead_location # Use same lead model and location as original df.
        interp_df['lead_model'] = np.repeat(group.loc[group.index[small_gap_start_inds], 'lead_model'].values, gap_sizes-1)
        interp_df['pt_id'] = pt_id
        interp_df['source_file'] = 'interp' # Denote filled rows as interpolated so we know they aren't real data.
        interp_df['interpolated'] = True
        return interp_df

def threshold_outliers(group: pd.DataFrame, cols_to_fill: list, threshold: float=((2**32)-1)/60) -> pd.DataFrame:
    """
    Replace outliers in the group DataFrame with NaN. These may then be left empty or optionally interpolated in a supplemental step.
    
    Parameters:
        group (pd.DataFrame): Input DataFrame with potential outliers.
        cols_to_fill (list): List of column names to fill outliers in.
        threshold (float, optional): Threshold to define the outliers. Default is max 32-bit int value divided by 60.
    
    Returns:
        pd.DataFrame: DataFrame with outliers filled.
    """
    new_cols = [(col[:-4] if col.endswith('_raw') else col) + '_threshold' for col in cols_to_fill]
    result = pd.DataFrame(index=group.index, columns=new_cols)
    result[new_cols] = group[cols_to_fill].values.copy()  # Copy original values to result DataFrame
    for col, new_col in zip(cols_to_fill, new_cols):
        mask_over_threshold = group[col] >= threshold
        result.loc[mask_over_threshold, new_col] = np.nan
    return result

def fill_outliers_OvER(group: pd.DataFrame, cols_to_fill: list):
    """
    Replace outliers in the group DataFrame caused by overvoltage readings. When the Percept device records a LFP value above its
    acceptable range, it places the maximum integer value in its place. Then, when the 10 minute interval is averaged,
    the abnormally high value dominates the average and causes non-physical outliers in the data. When multiple overages
    are observed in a single 10 minute interval, the outlier is even higher. Here, we estimate how many overages were
    recorded during each 10 minute interval, then remove them and recalculate the averaged LFP without the abnormal values.
    The overvoltage recordings may be caused by movement artifacts or something else.

    Parameters:
        - group (pd.DataFrame): DataFrame containing contiguous LFP data, potentially with outliers and holes.
        - cols_to_fill (list): List of column names to fill outliers in.

    Returns:
        - pd.DataFrame: DataFrame with outliers filled and containing number of overages in each cell.
    """
    new_filled_cols = [(col[:-4] if col.endswith('_raw') else col) + '_OvER' for col in cols_to_fill]
    new_num_overages_cols = [(col[:-4] if col.endswith('_raw') else col) + '_num_overages' for col in cols_to_fill]
    result_df = pd.DataFrame(index=group.index, columns=new_filled_cols + new_num_overages_cols)
    n = 60 # Number of samples per 10 minute average
    v = 2**32 - 1

    for col, new_filled_col, new_num_overages_col in zip(cols_to_fill, new_filled_cols, new_num_overages_cols):
        data = group[col].values
        num_overages = data // (v/n) # Estimate how many voltage overages we had during each 10 minute interval

        # If all samples within the interval are overages, place a NAN in. This will be filled in later when the missing values are filled.
        # This edge case never actually happens in our dataset, but we handle it just in case.
        valid_mask = num_overages < n
        corrected_data = np.empty_like(data, dtype=float)
        corrected_data[valid_mask] = (n * data[valid_mask] - v * num_overages[valid_mask]) / (n - num_overages[valid_mask])
        corrected_data[~valid_mask] = np.nan
        # print(corrected_data)

        result_df[new_filled_col] = corrected_data
        result_df[new_num_overages_col] = num_overages

    return result_df

def interpolate_holes(group: pd.DataFrame, cols_to_fill: list, max_gap: int=12) -> pd.DataFrame:
    """
    Fill missing values (NaNs) in the specified columns of the group DataFrame using PCHIP interpolation, for gaps up to max_gap size.
    
    Parameters:
        group (pd.DataFrame): Input DataFrame with missing values (NaNs).
        cols_to_fill (list): List of column names to fill.
        max_gap (int, optional): Maximum gap size to fill. Default is 12 (2 hours).
    
    Returns:
        pd.DataFrame: DataFrame with missing values filled in the specified columns.
    """
    new_cols = [col + '_interpolate' for col in cols_to_fill]
    filled_df = pd.DataFrame(index=group.index, columns=new_cols)
    filled_df[new_cols] = group[cols_to_fill].values.copy()  # Copy original values to filled DataFrame
    for new_col in new_cols:
        # Identify NaN indices
        nan_indices = np.where(filled_df[new_col].isna())[0]
        not_nan_indices = np.where(filled_df[new_col].notna())[0]
        valid_values = filled_df.loc[filled_df.index[not_nan_indices], new_col].values

        if len(not_nan_indices) < 2: # Not enough data to interpolate
            filled_df[new_col] = np.nan
            continue
        if len(nan_indices) == 0: # Nothing to interpolate
            continue

        # Create the PCHIP interpolator
        interpolator = PchipInterpolator(not_nan_indices, valid_values)
        gaps = np.split(nan_indices, np.where(np.diff(nan_indices) != 1)[0] + 1)
        if len(filled_df) - 1 in gaps[-1]:
            gaps = gaps[:-1]
        if (len(gaps) > 0) and (0 in gaps[0]):
            gaps = gaps[1:]

        for gap in gaps:
            if len(gap) <= max_gap:
                filled_df.loc[filled_df.index[gap], new_col] = interpolator(gap)
        
    return filled_df

def round_time_up(t: dttime, num_secs=600) -> dttime:
    total_seconds = t.hour * 3600 + t.minute * 60 + t.second
    rounded_seconds = (total_seconds + (num_secs-1)) // num_secs * num_secs
    if rounded_seconds >= 86400:
        return dttime(0, 0, 0)
    else:
        rounded_hour = rounded_seconds // 3600
        rounded_minute = (rounded_seconds % 3600) // 60
        return dttime(rounded_hour, rounded_minute, 0)

def gaussian_smooth(data: pd.Series, sigma: float=3.0) -> pd.Series:
    """
    Apply Gaussian smoothing to a pandas Series.
    
    Parameters:
        data (pd.Series): Input data to smooth.
        sigma (float, optional): Standard deviation for Gaussian kernel. Default is 3.0.
    
    Returns:
        pd.Series: Smoothed data.
    """
    if len(data) < 2:
        return data
    if data.isna().all():
        return pd.Series([np.nan] * len(data), index=data.index)
    interpolated = data.interpolate(method='linear', limit_direction='both')
    smoothed = gaussian_filter1d(interpolated.values, sigma=sigma, mode='wrap')
    return pd.Series(smoothed, index=data.index)