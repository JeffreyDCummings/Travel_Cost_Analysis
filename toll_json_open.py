""" Analyze the best route for a given city to city road trip based on an input hourly time
 rate, whether or not cash or tags are being used for tolls, cost of gas, and car maintenance
 and depreciation costs. Calculation parameters are also output and an option to display
 the google maps of the best route is given."""

import webbrowser
import json
import sys

HOURLY_RATE = 18
JSON_FILE = "ChicagoILIndianapolisIN.json"

#With tags, set to 0, with cash, set to 1
CASH_OR_TAGS = 0
GAS_PRICE = 3
# Cost per mile to drive a car (maintenance + depreciation)
MAINTENANCE = 0.395

#Read json file output by tollguru_api.pyi
def jsonread(input_file):
    """ Read input json file and output to a dataframe. """
    with open(input_file, "r") as read_file:
        return json.load(read_file)

def correct_nones(check):
    """ Check if a value is None, if so, return 10000, else return value. """
    if check is None:
        return 100000
    return check

def drive_time(data, route_num):
    """ For cost calculations, take the output route duration and return it in hours. """
    duration_hours = 0
    duration = data["routes"][route_num]["summary"]["duration"]["text"].split()
    if duration[1] == 'd':
        duration_hours += int(duration[0]) * 24
    if duration[1] == 'h':
        duration_hours += int(duration[0])
    if duration[1] == 'min':
        duration_hours += int(duration[0])/60
    if len(duration) > 3:
        if duration[3] == 'h':
            duration_hours += int(duration[2])
        if duration[3] == 'min':
            duration_hours += int(duration[2])/60
    if len(duration) > 5:
        if duration[5] == 'min':
            duration_hours += int(duration[4])/60
    return duration_hours

def calc_cost(data):
    """ Calculates cost of extra time and cost of extra tolls, based on input hourly rate and\
     whether or not cash or tags will be used for tolls."""
    toll_cost, time_cost, total_costs, cashpossible, licensepossible = [], [], [], 1, 1
    for route_num in range(len(data["routes"])):
        route_costs_dict = data["routes"][route_num]["costs"]
        drive_costs = route_costs_dict["fuel"]*GAS_PRICE/3 +\
         float(data["routes"][route_num]["summary"]["distance"]["text"].split()[0]) * MAINTENANCE
        if data["routes"][route_num]["summary"]["hasTolls"]:
            if CASH_OR_TAGS == 0:
                route_costs_dict["tag"] = correct_nones(route_costs_dict["tag"])
                toll_cost.append(route_costs_dict["tag"])
            else:
                route_costs_dict["cash"] = correct_nones(route_costs_dict["cash"])
                route_costs_dict["licensePlate"] = correct_nones(route_costs_dict["licensePlate"])
                toll_cost.append(min(route_costs_dict["cash"], route_costs_dict["licensePlate"]))
                if route_costs_dict["cash"] > 50000:
                    cashpossible = 0
                if route_costs_dict["licensePlate"] > 50000:
                    licensepossible = 0
        else:
            toll_cost.append(0)
        duration_hours = drive_time(data, route_num)
        time_cost.append(round(duration_hours * HOURLY_RATE, 2))
        total_costs.append(round(toll_cost[route_num]+time_cost[route_num]+drive_costs, 2))
    return toll_cost, time_cost, total_costs, cashpossible, licensepossible

def findminindex(total_costs):
    """ Determines which route number is the cheapest. """
    mincost = min(total_costs)
    return total_costs.index(mincost)

def output(minindex, toll_cost, time_cost, cashpossible, licensepossible, total_costs):
    """ Outputs information to screen based on setup and results."""
    if CASH_OR_TAGS == 0:
        print("Route "+str(minindex)+" is best for an hourly rate of $"\
         +str(HOURLY_RATE)+" and using tags.")
    else:
        if cashpossible == 0 and licensepossible == 0:
            print("A toll free, cash only, or license plate based route is unavailable.")
            sys.exit()
        elif cashpossible == 1:
            if licensepossible == 0:
                print("Route "+str(minindex)+" is best for an hourly rate of $"\
                 +str(HOURLY_RATE)+" and using cash.")
            if licensepossible == 1:
                print("Route "+str(minindex)+" is best for an hourly rate of $"+str(HOURLY_RATE)+\
                " and using cash AND license plates")
                print("IMPORTANT: License plate registration may be necessary.")
        elif licensepossible == 1:
            print("Route "+str(minindex)+" is best for an hourly rate of $"+str(HOURLY_RATE)+\
             " and using license plates")
            print("IMPORTANT: License plate registration may be necessary.")
    print("Extra Toll and Fuel Costs: $"+str(toll_cost[minindex])+\
    " (Relative to the Cheapest Route Based on Tolls and Fuel).")
    print("Extra Time Costs: $"+str(round(time_cost[minindex]-min(time_cost), 2))+\
     " (Relative to the Fastest Route).")
    print("Total Travel Costs: $"+str(total_costs[minindex]))

def googlemapscall(minindex, data):
    """ Draws googlemaps url from json file for the best route and asks whether or not
      the user wants to display the map. """
    route = data["routes"][minindex]["summary"]["url"]
    answer = input("Would you like displaying Route "+str(minindex)+" in Google Maps? Y/N\n")
    if answer in ["y", "Y", "Yes", "yes", "Sure", "sure", "OK", "ok"]:
        webbrowser.open(route, new=2, autoraise=True)
    else:
        print("Enjoy Your Trip")

def main():
    """ The Main Program Series."""
    data = jsonread(JSON_FILE)
    toll_cost, time_cost, total_costs, cashpossible, licensepossible = calc_cost(data)
    minindex = findminindex(total_costs)
    if data["routes"][minindex]["summary"]["diffs"]["fastest"] == 0:
        print("Route "+str(minindex)+" is both the fastest route and the cheapest route.")
        print("Total Travel Costs: $"+str(total_costs[minindex]))
    else:
        output(minindex, toll_cost, time_cost, cashpossible, licensepossible, total_costs)
    googlemapscall(minindex, data)

main()
