import pandas as pd
from matplotlib import pyplot as plt

file_path = "/Users/Timon/Documents/Houston/data_standardization/my_parser_scripts/NBU_poster/Report_Json_Session_Report_20250410T170729/BrainSenseTimeDomain_2025-04-10 22:21:50 - 22:41:02.csv"

df = pd.read_csv(file_path)

df_ch_0_2_l = df.query("Channel == 'ZERO_TWO_LEFT'")[["Time", "Value"]]
df_ch_0_2_l.to_csv("/Users/Timon/Documents/Houston/data_standardization/my_parser_scripts/NBU_poster/Report_Json_Session_Report_20250410T170729/ch_0_2_l.csv", index=False)
df_ch_0_2_l["Time"] = pd.to_datetime(df_ch_0_2_l["Time"], format="mixed")
df_ch_0_2_l["Value"].to_csv("/Users/Timon/Documents/Houston/data_standardization/my_parser_scripts/NBU_poster/Report_Json_Session_Report_20250410T170729/ch_0_2_l_values.csv", index=False)

plt.figure()
plt.plot(df_ch_0_2_l["Time"], df_ch_0_2_l["Value"])
plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Channel ZERO_TWO_LEFT")
plt.show()