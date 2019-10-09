"""This script reads in the Toll Guru API JSON files that are within its directory
and performs the comparitive calculations and outputs the results to two csv files."""
import json
import os.path
import csv
import numpy as np
import pandas as pd
from city_set import city_set, extract_count

np.set_printoptions(suppress=True)

HOURLY_RATE_MAXOUT = 30
HOURLY_RATE_MAXCALC = 2500
GAS_PRICE = 3
# Cost per mile to drive a car (maintenance + depreciation)
MAINTENANCE = 0.395

TOLL_CALC_OUT = "tollcalcdata.csv"
ONE_ROUTE_OUT = "onlyoneroute.csv"
POLYLINE_OUT = "fastest_polylines.csv"


def csvwrite(dataframe):
    """Create csv output and then append city dataframe to output to csv file. """
    if os.path.isfile(TOLL_CALC_OUT):
        headcheck = False
    else:
        headcheck = True
    with open(TOLL_CALC_OUT, 'a') as out_file:
        dataframe.to_csv(out_file, header=headcheck)


def jsonread(input_file):
    """ Load the JSON file output by the Toll Guru API and return it."""
    with open(input_file, "r") as read_file:
        return json.load(read_file)

def correct_nones(check):
    """ Check if a value is None, if so, return 10000, else return value. """
    if check is None:
        return 100000
    return check

def check_different_costs(data, length):
    """ Check all available route costs, including replacing None values when a payment\
     type is unavailable. """
    costlistcashlicense, costlisttag, routeinfo, cash_license_works = [], [], [], 0
    for route_num in range(min(6, length)):
# Extracts dictionary of costs and calculates gas and average maintenance costs of the entire trip.
        route_costs_dict = data["routes"][route_num]["costs"]
        drive_costs = route_costs_dict["fuel"]*GAS_PRICE/3 +\
         float(data["routes"][route_num]["summary"]["distance"]["text"].split()[0]) * MAINTENANCE
        if data["routes"][route_num]["summary"]["hasTolls"]:
# If the cash, licenseplate, or tag options are set to None for the route, they are changed
# to 10000, which allows the subsequent calculations to continue but makes the resulting
# route costs unviable in comparison to any route that does not lack the option.
            route_costs_dict["cash"] = correct_nones(route_costs_dict["cash"])
            route_costs_dict["licensePlate"] = correct_nones(route_costs_dict["licensePlate"])
            route_costs_dict["tag"] = correct_nones(route_costs_dict["tag"])
            if route_costs_dict["cash"] < 50000 or route_costs_dict["licensePlate"] < 50000:
                cash_license_works = 1
            costlistcashlicense.append(min(route_costs_dict["cash"],\
             route_costs_dict["licensePlate"])+drive_costs)
            costlisttag.append(route_costs_dict["tag"]+drive_costs)
        else:
            costlistcashlicense.append(drive_costs)
            costlisttag.append(drive_costs)
            cash_license_works = 1
    return costlistcashlicense, costlisttag, routeinfo, cash_license_works

def normalize_costs(costlistcashlicense, costlisttag):
    """ Reset minimum route cost, both with tags or cash/license, to zero. """
    costcashlicensemin = min(costlistcashlicense)
    costlistcashlicense = [round(route_num-costcashlicensemin, 2) for\
     route_num in costlistcashlicense]
    costtagmin = min(costlisttag)
    costlisttag = [round(route_num-costtagmin, 2) for route_num in costlisttag]
    return costlistcashlicense, costlisttag

def fast_route_availability(data, route_num):
    """ Checks if the fast route has tolls, and whether or not cash/license pay is available. """
    route_costs_dict = data["routes"][route_num]["costs"]
    if data["routes"][route_num]["summary"]["hasTolls"] and (route_costs_dict["cash"] >\
     50000 and route_costs_dict["licensePlate"] > 50000):
        return [route_num, False]
    return [route_num, True]

def calc_cost(data):
    """Takes the input JSON file data, calculates the cost of extra time and the cost of
    tolls, and returns the appropriate information on the available routes. """
    cheaproutespeed = 5000
    fastroutecost = 5000
    extracost = []
    length = len(data["routes"])
    costlistcashlicense, costlisttag, routeinfo, cash_license_works =\
     check_different_costs(data, length)
    costlistcashlicense, costlisttag = normalize_costs(costlistcashlicense, costlisttag)
    for route_num in range(min(6, length)):
        route_time = data["routes"][route_num]["summary"]["diffs"]["fastest"]
        routeinfo.extend(["Route "+str(route_num), costlisttag[route_num], route_time,\
         round(data["routes"][route_num]["costs"]["fuel"]*GAS_PRICE/3, 2),\
         data["routes"][route_num]["summary"]["url"]])
        if costlisttag[route_num] == 0 and route_time == 0:
            fastroute = fast_route_availability(data, route_num)
            return 0, route_num, fastroute, ["Route "+str(route_num),\
             str(costlisttag[route_num]), str(route_time),\
             str(round(data["routes"][route_num]["costs"]["fuel"]*GAS_PRICE/3, 2)),\
             str(data["routes"][route_num]["summary"]["url"])],\
              data["routes"][route_num]["polyline"], cash_license_works
    # Check Route number of cheapest
        if costlisttag[route_num] == 0 and route_time < cheaproutespeed:
            cheaproute = route_num
            cheaproutespeed = route_time
        if route_time == 0 and costlisttag[route_num] < fastroutecost:
            fastroute = fast_route_availability(data, route_num)
            fastroutecost = costlisttag[route_num]
            fastroute_polyline = data["routes"][route_num]["polyline"]
        for hourlyrate in range(HOURLY_RATE_MAXCALC+1):
            extracost.append([round(costlisttag[route_num]+route_time*hourlyrate/60,\
             2), hourlyrate, route_num, 0])
            extracost.append([round(costlistcashlicense[route_num]+route_time*hourlyrate/60,\
             2), hourlyrate, route_num, 1])
    return extracost, cheaproute, fastroute, routeinfo, fastroute_polyline, cash_license_works


def find_best_route(extracostnp, hourlyrate, tags_or_cash):
    """Take the numpy array of each route's cost calculations at a given hourlyrate and
     and for a selected tags_or_cash and returns the minimum cost route number. """
    extracostnptemp = extracostnp[(extracostnp[:, 1] == hourlyrate) & (extracostnp[:, 3]\
     == tags_or_cash)]
    return np.unravel_index(extracostnptemp[:, 0].argmin(), extracostnptemp[:, 0].shape)[0]


def findminhourlyrate(ratemin, extracostnp, tags_or_cash, maxoutput, fastroute):
    """ This is called after the HOURLY_RATE_MAXOUT is reached, and if the fastest
     route has not yet become the best route, this function continues until it does (or
     until it reaches the much larger HOURLY_RATE_MAXCALC) and returns that hourlyrate. """
    if ratemin == 0:
        if not fastroute[1] and tags_or_cash == 1:
            return "NaN"
        for hourlyrate in range(HOURLY_RATE_MAXOUT, HOURLY_RATE_MAXCALC+1):
            best = find_best_route(extracostnp, hourlyrate, tags_or_cash)
            if best == fastroute[0]:
                return hourlyrate
        return maxoutput
    if fastroute[1] or tags_or_cash == 0:
        return ratemin
    return "NaN"

def findminindex(extracostnp, cheaproute, fastroute, routeinfo, cash_license_works):
    """ Determines which route number is the cheapest for each input hourlyrate up to
    HOURLY_RATE_MAXOUT and count which route type is the best for each city to city route. """
    # rate_count is a list where index 0 represents the number of times the fastest route
    # is best, index 2 represents the number of times the cheapest route is best, and index
    # 1 represents the number of times neither are best.
    route_count = [0, 0, 0]
    # ratemin is a list where ratemin[tags_or_cash] is the minimum hourlyrate where the
    # fastest route becomes the best route.
    ratemin = [0, 0]
    coltitles, mincost = [], []
    keyword = ["tags", "cash"]
    for hourlyrate in range(HOURLY_RATE_MAXOUT+1):
        for tags_or_cash in range(2):
            coltitles.append("$"+str(hourlyrate)+"/hr with "+keyword[tags_or_cash])
            if (cash_license_works == 1 or tags_or_cash == 0):
                best = find_best_route(extracostnp, hourlyrate, tags_or_cash)
                mincost.append("Route "+str(best))
                if best == cheaproute:
                    route_count[2] += 1
                    continue
                if best == fastroute[0]:
                    route_count[0] += 1
                    if ratemin[tags_or_cash] == 0:
                        ratemin[tags_or_cash] = hourlyrate
                    continue
                if best not in (cheaproute, fastroute[0]):
                    route_count[1] += 1
            else:
                mincost.append("NaN")

# If the minimum hourly rate where the fastest route becomes the best route has not yet
# been reached, this continues until either it does or it reaches HOURLY_RATE_MAXCALC.
    for tags_or_cash in range(2):
        max_reached = [str(HOURLY_RATE_MAXCALC)+"+", "NaN"]
        ratemin[tags_or_cash] = findminhourlyrate(ratemin[tags_or_cash], extracostnp,\
         tags_or_cash, max_reached[tags_or_cash], fastroute)
    while len(routeinfo) < 30:
        routeinfo.extend(["NaN", "NaN", "NaN", "NaN", "NaN"])
    mincost.extend(route_count+ratemin+routeinfo)
    coltitles.extend(["Fastest Best", "Neither Best", "Cheapest Best", "Min Rate Tags",\
     "Min Rate Cash"])
    for route_num in range(6):
        coltitles.extend(["Route #", "Cheapest "+str(route_num), "Fastest "+str(route_num),\
         "Fuel "+str(route_num), "Google Maps URL "+str(route_num)])
    return mincost, coltitles

def onlyoneroute_write(data, cities_string, fastroute, routeinfo, fastroute_polyline):
    """ Write routes with only one viable route to the ONE_ROUTE_OUT file. """
    onlyoneroute = [cities_string, "Route "+str(fastroute[0])+\
     " is both the fastest route and the cheapest route",\
     str(data["routes"][fastroute[0]]["summary"]["hasTolls"]),\
     fastroute[1]]
    onlyoneroute.extend(routeinfo)
    onlyoneroute.extend([str(fastroute_polyline)])
    with open(ONE_ROUTE_OUT, 'a', newline='') as myfile:
        if os.stat(ONE_ROUTE_OUT).st_size == 0:
            write_file = csv.writer(myfile)
            write_file.writerow(["Cities", "Best Route", "Has Tolls",\
             "Cash/License Plates Available", "Route Number", "Cheapest",\
             "Fastest", "Fuel", "url", "Polyline"])
        write_file = csv.writer(myfile)
        write_file.writerow(onlyoneroute)

def write_output(mincost, coltitles, cities_string):
    """Create/return the dataframe including the calculations and also write it to csv. """
    mincostframe = pd.DataFrame([mincost], columns=coltitles, index=[cities_string])
    csvwrite(mincostframe)
    return mincostframe

def write_polyline(mincostframe, cities_string, fastroute_polyline):
    """Write the entire fast route information to POLYLINE_OUT."""
    polyline = [cities_string, str(mincostframe.iloc[0]["Min Rate Tags"]),\
     str(mincostframe.iloc[0]["Min Rate Cash"]), str(fastroute_polyline)]
    with open(POLYLINE_OUT, 'a', newline='') as myfile:
        if os.stat(POLYLINE_OUT).st_size == 0:
            write_file = csv.writer(myfile)
            write_file.writerow(["Cities", "Min Rate Tags",\
             "Min Rate Cash", "Polyline"])
        write_file = csv.writer(myfile)
        write_file.writerow(polyline)

def main():
    """ Main Program script, which calls the appropriate functions and appends the
     route data to two csv files for the two subclasses of routes."""
    citylist, length = extract_count("top35.csv")

    if os.path.exists(TOLL_CALC_OUT):
        os.remove(TOLL_CALC_OUT)
    if os.path.exists(ONE_ROUTE_OUT):
        os.remove(ONE_ROUTE_OUT)
    if os.path.isfile(POLYLINE_OUT):
        os.remove(POLYLINE_OUT)

    for start in range(length-1):
        for stop in range(start+1, length):
            start_city, stop_city = city_set(citylist, start, stop)
            cities_string = start_city+" to "+stop_city
            cities_file = start_city.replace(" ", "").replace(",", "") +\
             stop_city.replace(" ", "").replace(",", "")+".json"
            if os.path.isfile(cities_file):
                print(cities_file)
                data = jsonread(cities_file)
                extracost, cheaproute, fastroute, routeinfo, fastroute_polyline,\
                 cash_license_works = calc_cost(data)
# If there are meaningful route comparisons, extracost is a list, if there is only one
# valid route, the returned extracost is 0, an int.
                if isinstance(extracost, int):
                    onlyoneroute_write(data, cities_string, fastroute, routeinfo,\
                     fastroute_polyline)
                else:
                    mincost, coltitles = findminindex(np.array(extracost), cheaproute,\
                     fastroute, routeinfo, cash_license_works)
                    mincostframe = write_output(mincost, coltitles, cities_string)
                    write_polyline(mincostframe, cities_string, fastroute_polyline)
main()
