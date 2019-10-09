"""Call this to clean the list of JSON files of ones that are empty due to either
errors or too many calls to the toll_guru API. """

import json
import os

os.system('ls -1 *.json > jsonlist.txt')

list_array = []

with open('jsonlist.txt', 'r') as file:
    for line in file:
        list_array.append(line.rstrip())

for file_name in list_array:
    jsonfile = file_name
    with open(jsonfile, "r") as read_file:
        out = json.load(read_file)
    try:
        if out["message"] == "Too Many Requests" or out["message"] == "Limit Exceeded":
            os.system('rm '+file_name)
            print("Deleted "+file_name)
    except:
        try:
            if out["error"] == "Point not found":
                os.system('rm '+file_name)
                print("Deleted "+file_name)
        except:
            continue
