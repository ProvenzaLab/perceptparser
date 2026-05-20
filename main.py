from percept_parser import percept
import os

filename = "Report_Json_Session_Report_20250410T170448.json"

parser = percept.PerceptParser(filename, anonymize=True, out_path='anonymized')
parser.parse_all(plot=True)
