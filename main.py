from percept import PerceptParser
from plotter import brain_sense_lfp_plot, lfptrendlog_plot, brainsense_timedomain_plot
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from scipy import signal
import numpy as np
import pandas as pd


filename = "Report_Json_Session_Report_20250411T165744.json"
# filename = "Report_Json_Session_Report_20250425T080622.json"
# filename = "Report_Json_Session_Report_20240819T165729.json"
filename = "Report_Json_Session_Report_20250410T170448.json"
#filename = "Report_Json_Session_Report_20250604T095957.json"

PLOT_BRAINSENSE_LFP = True
PLOT_LFP_TREND_LOGS = True
PLOT_BRAINSENSE_TIME_DOMAIN = True

parser = PerceptParser(filename)
df_out = parser.read_indefinite_streaming()
df_out.to_csv("IndefiniteStreaming.csv", index=True)
df_out = pd.read_csv("IndefiniteStreaming.csv", parse_dates=["Time"], index_col="Time")

brainsense_timedomain_plot(df_out, parser, out_path="plots_is", indefinite_streaming=True)

df_bs_td, df_files_missing_entries = parser.read_brainsense_timedomain()


if PLOT_BRAINSENSE_TIME_DOMAIN:
    brainsense_timedomain_plot(df_bs_td, parser, out_path="plots", indefinite_streaming=False)

df_brainsense_lfp = parser.parse_brain_sense_lfp()

if df_brainsense_lfp.empty:
    print("No BrainSenseLfp data found in the JSON file.")
elif PLOT_BRAINSENSE_LFP:
    brain_sense_lfp_plot(df_brainsense_lfp, parser, out_path="plots")


df_trend_logs = parser.parse_lfp_trend_logs()
if df_trend_logs.empty:
    print("No LFPTrendLogs data found in the JSON file.")
elif PLOT_LFP_TREND_LOGS:
    lfptrendlog_plot(df_trend_logs, parser, path_out="plots")


