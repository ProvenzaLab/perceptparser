from percept_parser import percept
import os
import pandas as pd

df_paths = pd.read_csv("TRBD Programming Videos(TRBD001) (1).csv", sep=";").iloc[1:].query("Location != 'NBU'")
path_TRBD001 = "/mnt/datalake/data/TRBD-53761/TRBD001/LFP/R"

for file in df_paths["Path Neural"]:
    
    filename = os.path.join(path_TRBD001, file)
    parser = percept.PerceptParser(filename)
    dir_name = os.path.join("Jamail_rs_trbd001", os.path.basename(filename)[:-len(".json")])
    
    parser.parse_all(out_path=dir_name, plot=True)

# print(self.js["DeviceInformation"]["Initial"]['DeviceDateTime'])
# print(self.js["DeviceInformation"]["Final"]['DeviceDateTime'])
# print(self.js["SessionDate"])
# print(self.js["SessionEndDate"])
# FirstPackageDateTimes