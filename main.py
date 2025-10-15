from percept_parser import percept
from percept_parser.stim_settings import PatientStimSettingHistory, get_true_lead_location
import os
import json
from percept_parser.utils import chronic_lfp_from_json
from tqdm import tqdm
import numpy as np
import pandas as pd

filename = "Report_Json_Session_Report_20250410T170448.json"
filename = "/Users/Timon/Downloads/Report_Json_Session_Report_20250626T181946.json"
filename = '/Users/Timon/Downloads/Report_Json_Session_Report_20250709T102453 (1).json'
filename = "Report_Json_Session_Report_20250411T165744.json"

filename = "Report_Json_Session_Report_20240819T165729.json"

filename = "Report_Json_Session_Report_20250425T080622.json"
filename = "/Volumes/datalake/PerceptOCD-48392/009/LFP/R/Report_Json_Session_Report_009_09T164922_20240826T194325.json"
filename = "/Volumes/datalake/PerceptOCD-48392/011/LFP/R/Report_Json_Session_Report_011_04_20240104T151637.json"
filename = "/Volumes/datalake/PerceptOCD-48392/011/LFP/R/Report_Json_Session_Report_011_04_20240104T151236.json"
with open(filename, 'r') as f:
    js = json.load(f)

# Extract raw stim and LFP data from JSON to dataframe.
raw_df = chronic_lfp_from_json(js, filename)
stim_settings_hist = PatientStimSettingHistory(filename, None, None)
raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"])

# Assign stim settings to each row in the dataframe
raw_df[['left_stim_anodes', 'right_stim_anodes',
        'left_stim_cathodes', 'right_stim_cathodes',
        'left_sensing_contacts', 'right_sensing_contacts']] = 'Unknown'

pt_id = None

#pt_stim_settings = stim_settings_dict[pt_id]
for i, row in raw_df.iterrows():
    if row["timestamp"] > pd.Timestamp("2025-07-09", tz='UTC'):
        print("check here")

    left_settings, right_settings, during_session = stim_settings_hist.get_settings_at_time(row['timestamp'])
    # check if timestamp is larger than 2025-07-09

    if not during_session:
        for hem, settings in zip(['left', 'right'], [left_settings, right_settings]):
            if (settings is not None):# and (get_true_lead_location(row[f'{hem}_lead_location'], pt_id, hem).lower() == 'vc/vs'):
                cathodes = [c.lower().removeprefix('fourelectrodes_').removeprefix('sensight_') for c in settings.stim_cathodes]
                anodes = [a.lower().removeprefix('fourelectrodes_').removeprefix('sensight_') for a in settings.stim_anodes]
                raw_df.at[i, f'{hem}_stim_cathodes'] = ', '.join(cathodes)
                raw_df.at[i, f'{hem}_stim_anodes'] = ', '.join(anodes)
                raw_df.at[i, f'{hem}_sensing_contacts'] = settings.sensing_contacts
                raw_df.at[i, f'{hem}_group_name'] = settings.group_name
                raw_df.at[i, f'{hem}_settings_filename'] = str(settings.filename)
                raw_df.at[i, f'{hem}_frequency'] = settings.frequency
                raw_df.at[i, f'{hem}_pulse_width'] = settings.pulse_width
                raw_df.at[i, f'{hem}_suspend_amplitude'] = settings.suspend_amplitude
                raw_df.at[i, f'{hem}_sensing_frequency'] = settings.sensing_frequency
                raw_df.at[i, f'{hem}_lower_amplitude'] = settings.lower_amplitude
                raw_df.at[i, f'{hem}_upper_amplitude'] = settings.upper_amplitude
            else:
                raw_df.at[i, f'{hem}_stim_cathodes'] = 'Unknown'
                raw_df.at[i, f'{hem}_stim_anodes'] = 'Unknown'
                raw_df.at[i, f'{hem}_sensing_contacts'] = 'Unknown'
                raw_df.at[i, f'{hem}_group_name'] = 'Unknown'
                raw_df.at[i, f'{hem}_settings_filename'] = 'Unknown'
                raw_df.at[i, f'{hem}_frequency'] = np.nan
                raw_df.at[i, f'{hem}_pulse_width'] = np.nan
                raw_df.at[i, f'{hem}_suspend_amplitude'] = np.nan
                raw_df.at[i, f'{hem}_sensing_frequency'] = np.nan
                raw_df.at[i, f'{hem}_lower_amplitude'] = np.nan
                raw_df.at[i, f'{hem}_upper_amplitude'] = np.nan




parser = percept.PerceptParser(filename)
dir_name = os.path.basename(filename)[:-len(".json")]
parser.parse_all(out_path=dir_name)
