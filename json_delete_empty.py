"""Call this to clean the list of JSON files of ones that are empty due to either
errors or too many calls to the toll_guru API. """

import json
import os

def generate_local_json_list():
    """This function generates a list of the local json files and returns them as
     a list of strings. """
    os.system('ls -1 *.json > jsonlist.txt')

    json_list = []

    with open('jsonlist.txt', 'r') as file:
        for line in file:
            json_list.append(line.rstrip())
    return json_list

def check_delete(json_list):
    """This function takes in a json file name, or a list of them, and checks if they have any
    errors or features consistent with API-call failures.  If yes, these "empty" json files
    are deleted and printed to screen."""
    series_failed = False
    if isinstance(json_list, str):
        json_list = [json_list]
    for jsonfile in json_list:
        with open(jsonfile, "r") as read_file:
            out = json.load(read_file)
        try:
            if out["message"] == "Too Many Requests" or out["message"] == "Limit Exceeded"\
             or out["message"] == "Forbidden":
                os.system('rm '+jsonfile)
                print("Deleted "+jsonfile)
                series_failed = True
        except:
            try:
                if out["error"] == "Point not found":
                    os.system('rm '+jsonfile)
                    print("Deleted "+jsonfile)
                    series_failed = True
            except:
                continue
    return series_failed

def main():
    """The main program, which calls the two main functions when this program is run directly. """
    json_list = generate_local_json_list()
    _ = check_delete(json_list)

if __name__ == "__main__":
    main()
