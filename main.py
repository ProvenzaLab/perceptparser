from percept_parser import PerceptParser
import os

filename = "Report_Json_Session_Report_20250410T170448.json"
filename = "/Users/Timon/Downloads/Report_Json_Session_Report_20250626T181946.json"
filename = '/Users/Timon/Downloads/Report_Json_Session_Report_20250709T102453 (1).json'

parser = PerceptParser(filename)
dir_name = os.path.basename(filename)[:-len(".json")]
parser.parse_all(out_path=dir_name)
