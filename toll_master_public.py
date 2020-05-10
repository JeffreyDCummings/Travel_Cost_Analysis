"""Takes list of the largest cities and generates a series of start and stop locations and
inputs them into the Toll Guru API."""
import os.path
import sys
from city_set import city_set, extract_count
from toll_guru_api_series_public import par_set, api_call_write
from json_delete_empty import check_delete

# Read input cities CSV and LENGTH of city list.
CITIES, LENGTH = extract_count("top50edit.csv")

# Nested for loop for iterating over city list, producing all possible 2-city route
# combinations (order doesn't matter).
for x in range(LENGTH-1):
    for y in range(x+1, LENGTH):
        start, stop = city_set(CITIES, x, y)

# Remove spaces and commas from start and stop cities, which will be used to name
# output json file, etc.
        start0 = start.replace(" ", "").replace(",", "")
        stop0 = stop.replace(" ", "").replace(",", "")

# Check if json file already exists.  If yes, pass, if no, print route cities, setup
# API parameters, and call Toll Guru API.
        if os.path.isfile(start0+stop0+".json"):
            pass
        else:
            print(start+" to "+stop)
            Tolls_URL, headers, params = par_set(start, stop)
            api_call_write(Tolls_URL, headers, params, start0, stop0)
            series_failed = check_delete(start0+stop0+".json")
            if series_failed:
                print("API call failed or reached limit.")
                sys.exit()
