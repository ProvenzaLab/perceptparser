import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
from pathlib import Path
import re
import pickle
from matplotlib.backends.backend_pdf import PdfPages

pdf_ = PdfPages("NBU_stim_relation/all_visits_teed_right_all_channels.pdf")

df = pd.read_parquet("/mnt/labworlds/Provenza/NBUPipeline/TRBD001/2025-06-03/rawlfp/raw/TRBD001_2025-06-03-2025-06-05_timedomain_LFP.parq")

df_stim = pd.read_parquet("/scratch/timonmerk/get_data_NBU/perceptparser/trbd_chronic_df_stim_parameters.parq").query("pt_id == 'TRBD001'").copy()
df_stim["teed_right"] = 1000 * df_stim["right_frequency"] * (df_stim["right_pulse_width"] * (10 **-6)) * ((df_stim["stim_right"] * 10 **-3) ** 2)
df_stim["teed_left"] = 1000 * df_stim["left_frequency"] * (df_stim["left_pulse_width"] * (10 **-6)) * ((df_stim["stim_left"] * 10 **-3) ** 2)
# save only entries where teed_right or teed_left changed
# sort by timestamp first
df_stim = df_stim.sort_values("timestamp").reset_index(drop=True)
df_stim = df_stim[
        (df_stim["teed_right"].diff().abs() > 0)
        | (df_stim["teed_left"].diff().abs() > 0)
    ].copy()

base_dir = Path("/scratch/timonmerk/nbu_lfp_preproc/nbu_concat_single_channel")
pkl_files = sorted(base_dir.glob("spectra_highres_fooof_bandpower_*.pkl"))

VISIT_FILE_RE = re.compile(
    r"^spectra_highres_fooof_bandpower_(?P<channel>.+?)_visit(?P<visit>\d+)_(?P<start>\d{4}-\d{2}-\d{2})_to_(?P<end>\d{4}-\d{2}-\d{2})$"
)

for pkl_file in pkl_files:
    match = VISIT_FILE_RE.match(pkl_file.stem)
    if not match:
        print(f"Filename {pkl_file.name} does not match the expected pattern.")
        continue

    channel = match.group("channel")
    if "LEFT" in channel.upper():
        continue
 
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)

    visit = int(match.group("visit"))
    start_date = data["df_features"].index[0]
    end_date = data["df_features"].index[-1]

    df_visit_stim = df_stim[
        (df_stim["timestamp"] >= start_date) & (df_stim["timestamp"] <= end_date)
    ].copy()
    # get only entries where teed_right changed
    #df_visit_stim = df_visit_stim[df_visit_stim["teed_right"].diff().abs() > 0].copy()

    plt.figure(figsize=(10, 15))
    plt.subplot(4, 1, 1)
    plt.plot(df_visit_stim["timestamp"], df_visit_stim["teed_right"], marker='o', linestyle='-')
    plt.title(f"TEED Right for Channel {channel} Visit {visit}")
    plt.xlabel("Timestamp")
    plt.ylabel("TEED Right")
    plt.xticks(rotation=45)
    plt.subplot(4, 1, 2)
    plt.plot(df_visit_stim["timestamp"], df_visit_stim["teed_left"], marker='o', linestyle='-')
    plt.title(f"TEED Left for Channel {channel} Visit {visit}")
    plt.xlabel("Timestamp")
    plt.ylabel("TEED Left")
    plt.xticks(rotation=45)
    plt.subplot(4, 1, 3)
    plt.plot(df_visit_stim["timestamp"], df_visit_stim["stim_right"], marker='o', linestyle='-')
    plt.title(f"Stim Right for Channel {channel} Visit {visit}")
    plt.xlabel("Timestamp")
    plt.ylabel("Stim Right")
    plt.xticks(rotation=45)
    plt.subplot(4, 1, 4)
    plt.plot(df_visit_stim["timestamp"], df_visit_stim["stim_left"], marker='o', linestyle='-')
    plt.title(f"Stim Left for Channel {channel} Visit {visit}")
    plt.xlabel("Timestamp")
    plt.ylabel("Stim Left")
    plt.xticks(rotation=45)
    plt.tight_layout()
    pdf_.savefig(plt.gcf())
pdf_.close()