from percept_parser import percept
import os

#filename = "Report_Json_Session_Report_20250410T170448.json"
#filename = "/Users/Timon/Downloads/Report_Json_Session_Report_20250626T181946.json"
#filename = '/Users/Timon/Downloads/Report_Json_Session_Report_20250709T102453 (1).json'

filename = "/Volumes/datalake/TRBD-53761/TRBD002/LFP/R/Report_Json_Session_Report_20250807T092340.json"
filename = "/Volumes/datalake/TRBD-53761/TRBD002/LFP/R/Report_Json_Session_Report_20250807T092405.json"
filename = "/Volumes/datalake/TRBD-53761/TRBD002/LFP/R/Report_Json_Session_Report_20250807T092428.json"
filename = "/Volumes/datalake/TRBD-53761/TRBD002/LFP/R/Report_Json_Session_Report_20250807T092621.json"
filename = "/Volumes/datalake/TRBD-53761/TRBD002/LFP/R/Report_Json_Session_Report_20250807T092626.json"
filename = "/Volumes/datalake/TRBD-53761/TRBD002/LFP/R/Report_Json_Session_Report_20250807T092820.json"
filename = "/Volumes/datalake/TRBD-53761/TRBD002/LFP/R/Report_Json_Session_Report_20250807T092843.json"
filename = "/Users/Timon/Documents/perceive/Report_Json_Session_Report_20250410T170729.json"

parser = percept.PerceptParser(filename)
dir_name = os.path.basename(filename)[:-len(".json")]
out_path = os.path.join("NBU_poster", dir_name)
outpath = "test"
parser.parse_all(out_path=out_path, plot=True)