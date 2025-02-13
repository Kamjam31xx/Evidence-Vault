import argparse
import time
from datetime import datetime
from modules.database_utils import db_text_result_search, db_print_file_scan_data_rows

initialization_time = time.time()
initialization_datetime = datetime.now().isoformat()
print(f"[{initialization_datetime}] Started database search")

# Argument parser setup
parser = argparse.ArgumentParser(description="Perform STT on selected file or directory with multiple STT engines")
parser.add_argument('-t', '--action_t', type=str, help="Perform total match query")
parser.add_argument('-s', '--action_s', type=str, help="Perform sub-string query")
parser.add_argument('-f', '--action_f', type=str, help="Perform fuzzy query")
parser.add_argument("--case-sensitive", action="store_true", help="case-sensitive query")
parser.add_argument('--info', action='store_true', help="Display information per selected flag")
args = parser.parse_args()

input_string = None
function_ptr = None

if args.action_t:
    print("to-do")
elif args.action_s:
    input_string = args.action_s
    function_ptr = db_text_result_search
elif args.action_f:
    print("to-do")
else:
    print("Invalid entry")

results = None

if (input_string != None) and (function_ptr != None) :
    if not args.case_sensitive:
        input_string = input_string.lower()
    results = function_ptr(input_string)

search_duration = time.time() - initialization_time
num_matches = 0
if results != None :
    num_matches = len(results)
    db_print_file_scan_data_rows(results)
print(f"Found {num_matches} results for \"{input_string}\" in {round(search_duration, 5)} seconds.")