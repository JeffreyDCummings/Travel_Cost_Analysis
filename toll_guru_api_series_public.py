"""These functions setup the Toll Guru parameters and calls its API. """
import json
import requests

def par_set(start, stop):
    """ The par_set function takes the start and stop cities of the route and
    defines/returns the input API parameters. """
    tolls_url = 'https://dev.tollguru.com/beta00/calc/here'
    headers = {'Content-type': 'application/json',\
                'x-api-key': 'PUT YOUR TOLL GURU API-KEY HERE'}
    params = {'from': {'address': start},\
               'to': {'address': stop},\

              # A 2-axle standard auto is used, and this must be changed if a
              # commercial vehicle or motorcycle is being used.  See:
              # https://tollguru.com/developers/docs/#vehicle-types 
              # instead, for the appropriate key word of other vehicle types.
              'vehicleType': '2AxlesAuto',\

              # A unix time for departure is set to November 15th, 2019 at noon
              # UTC time.  If this is in the past, current time is used instead.
              'departure_time': '1573837200',\

              # DO NOT CHANGE 'fuelprice' here if you want to apply the json output to
              # toll_json_open.py or toll_analyzer.py.  These programs assume an input
              # cost of $3.00/gallon but take an input fuel price of their own to allow
              # adjustments of fuel price for a given route without having to call the API
              # again. 
              'fuelPrice': 3.00,\
              # The typical US city and highway fuel efficiencies for a 2-axle are used,
              # and do change these rates here if necessary for more personalized calculations.
              'fuelEfficiency': {'city': 23.4, 'hwy': 29.25}\
            }
    return tolls_url, headers, params

def api_call_write(tolls_url, headers, params, start0, stop0):
    """Toll Guru API is called with input parameters and the output is written to a json
    file named after the start/stop cities. """
    response = requests.post(tolls_url, json=params, headers=headers)

    json_data = response.json()

    outfile = open(start0+stop0+".json", 'w')
    json.dump(json_data, outfile, indent=4, sort_keys=True, ensure_ascii=True)
