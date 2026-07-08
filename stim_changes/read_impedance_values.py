import pandas as pd
import numpy as np
import os
import json
from tqdm import tqdm
from matplotlib import pyplot as plt

READ_ = False

if READ_:
    json_files = [f for f in os.listdir("/mnt/datalake/data/TRBD-53761/TRBD001/LFP/R") if f.endswith(".json")]

    impedance_values = []
    for json_file in tqdm(json_files):
        file_path = os.path.join("/mnt/datalake/data/TRBD-53761/TRBD001/LFP/R", json_file)
        with open(file_path, "r") as f:
            data = json.load(f)
            session_date = data["SessionDate"]
            if "Impedance" in data:
                num_entries = len(data["Impedance"])
                if num_entries > 0:
                    if num_entries > 1:
                        print(f"Warning: More than one impedance entry found in {json_file}. Using the first entry.")
                    hem_entries = data["Impedance"][0]["Hemisphere"]
                    for hem_entry in hem_entries:
                        hem_name = hem_entry["Hemisphere"]
                        imp = hem_entry["SessionImpedance"]["Monopolar"]
                        for imp_entry in imp:
                            imp_entry["SessionDate"] = session_date
                            imp_entry["Hemisphere"] = hem_name
                            imp_entry["type"] = "Monopolar"
                            impedance_values.append(imp_entry)
                        imp = hem_entry["SessionImpedance"]["Bipolar"]
                        for imp_entry in imp:
                            imp_entry["SessionDate"] = session_date
                            imp_entry["Hemisphere"] = hem_name
                            imp_entry["type"] = "Bipolar"
                            impedance_values.append(imp_entry)
    df_impedance = pd.DataFrame(impedance_values)
    df_impedance.to_csv("stim_changes/impedance_values.csv", index=False)

else:
    df_impedance = pd.read_csv("stim_changes/impedance_values.csv")
    df_impedance["Hemisphere"] = df_impedance["Hemisphere"].apply(lambda x: x.split(".")[1])
    df_impedance["Electrode2"] = df_impedance["Electrode2"].apply(lambda x: x.split(".")[1])
    df_impedance["Electrode1"] = df_impedance["Electrode1"].apply(lambda x: x.split(".")[1])

    df_r_bip = df_impedance.query("Hemisphere == 'Right' and type == 'Bipolar'")

    df_r_bip_filtered = df_r_bip.query(
        "Electrode1 == 'SenSight_0' and Electrode2.str.contains('2')",
        engine="python"
    )

    df_r_bip_filtered["Electrode2"] = df_r_bip_filtered["Electrode2"].apply(lambda x: x.split("_")[1])
    df_r_bip_filtered["Electrode1"] = df_r_bip_filtered["Electrode1"].apply(lambda x: x.split("_")[1])
    df_r_bip_filtered["SessionDate"] = pd.to_datetime(df_r_bip_filtered["SessionDate"])
    # sort by session date
    df_r_bip_filtered = df_r_bip_filtered.sort_values(by="SessionDate")
    df_r_bip_filtered["pair"] = df_r_bip_filtered.apply(lambda row: f"{row['Electrode1']}-{row['Electrode2']}", axis=1)

    plt.figure(figsize=(15, 10))
    for pair in df_r_bip_filtered["pair"].unique():
        pair_data = df_r_bip_filtered[df_r_bip_filtered["pair"] == pair]
        plt.plot(pd.to_datetime(pair_data["SessionDate"]), pair_data["ResultValue"], label=f"Pair {pair}", marker="o", linestyle="-", markersize=4)
    plt.xlabel("Session Date")
    plt.ylabel("Impedance (Ohms)")
    plt.title("Impedance Values Over Time for Each Electrode Pair (Right Hemisphere, Bipolar)")
    plt.legend()
    plt.savefig("stim_changes/impedance_values_over_time_bipolar_right.pdf")


    df_impedance["Contact"] = df_impedance["Electrode2"].apply(lambda x: x.split(".")[1])
    # Session date has form 2025-05-19T18:37:30Z, sort
    df_impedance["SessionDate"] = pd.to_datetime(df_impedance["SessionDate"])

df_impedance = df_impedance.sort_values(by="SessionDate")
plt.figure(figsize=(15, 10))
for contact in df_impedance["Contact"].unique():
    contact_data = df_impedance[df_impedance["Contact"] == contact]
    plt.plot(pd.to_datetime(contact_data["SessionDate"]), contact_data["ResultValue"], label=f"Contact {contact}")
plt.xlabel("Session Date")
plt.ylabel("Impedance (Ohms)")
plt.title("Impedance Values Over Time for Each Contact")
plt.legend()
plt.savefig("stim_changes/impedance_values_over_time.pdf")