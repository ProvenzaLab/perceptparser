from percept import PerceptParser
import os

filename = "Report_Json_Session_Report_20250425T080622.json"

parser = PerceptParser(filename)
dir_name = os.path.basename(filename)[:-len(".json")]
parser.parse_all(out_path=dir_name)
