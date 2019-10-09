""" Set of functions for reading and formatting lists of cities. """
import pandas as pd

def extract_count(citieslist):
    """ Reads in large population city list and returns this as list and its length. """
    cities = pd.read_csv(citieslist, delimiter=",")
    return cities, len(cities)

def city_set(cities, startnum, stopnum):
    """ For iterating through the city list, this creates and returns a properly
     formatted start city+state and stop city+state. """
    start = str(cities["city_ascii"][startnum]+", "+cities["state_id"][startnum])
    stop = str(cities["city_ascii"][stopnum]+", "+cities["state_id"][stopnum])
    return start, stop
