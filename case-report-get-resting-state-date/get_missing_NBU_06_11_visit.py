
import sys
# add the parent directory to the path so we can import percept_parser
sys.path.append("/scratch/timonmerk/get_data_NBU/perceptparser")
from percept_parser import percept
import os
import pandas as pd

path_TRBD001 = "/mnt/datalake/data/TRBD-53761/TRBD001/LFP/R"
file = "Report_Json_Session_Report_20250611T110521.json"  # LOUNGE, phone use 

filename = os.path.join(path_TRBD001, file)
parser = percept.PerceptParser(filename)
dir_name = os.path.join("/scratch/timonmerk/get_data_NBU/perceptparser/Jamail_rs_trbd001", os.path.basename(filename)[:-len(".json")])

parser.parse_all(out_path=dir_name, plot=True)

# /scratch/timonmerk/get_data_NBU/perceptparser/Jamail_rs_trbd001/Report_Json_Session_Report_20250611T110521/BrainSenseTimeDomain_2025-06-11_13-30-48_13-35-13.csv
# get first 2 minutes
