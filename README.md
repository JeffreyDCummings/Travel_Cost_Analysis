# Travel_Cost_Analysis
Analyze the full costs of multiple available city to city routes, and identify which is best under a given scenario.

Purpose of Program Set:

Long distance car travel is very common throughout the United States.  However, a true sense of the costs of traveling by car, and how
these costs vary throughout the different available routes from your starting point to your destination are rarely analyzed thoroughly. 
For example, in addition to fuel costs, how much costs are you putting towards maintenance and depreciation of your vehicle?  How much do
the available routes differ in terms of costs towards tolls (with or without a toll tag), and how much time do you save by paying for
these tolls?  How much is this saved time worth to you in a rate of dollars/hour?  When does the worth of your time make taking a given
toll worth the toll cost?  This set of programs analyses all available options and both outputs the best available routes, their 
corresponding total costs, and analyses the statistics of this comparison for over 500 common city to city travel routes in the United
States.  The details of each individual program and example output file are given below:

This set of programs analyses this by first calling the Toll Guru API (tollguru.com) and then for a specific starting and ending 
destination, toll_json_open.py performes the additional calculations and route comparisons and for a given fuel cost per gallon, 
maintenance cost per mile, and time cost per hour.  The best route is given with the corresponding cost information of this best route. 

Overview of programs:

toll_guru_api_series_public.py: This program is not run directly, but it is called by toll_master_public.py.  This is the public version 
of toll_guru_api_series.py, where my personal toll guru API key has been excluded.  A user will need to place their own API on line 10 for
the program to work.  This program also sets the necessary information of vehicle type (2-axle car is currently adopted), the start time
of the trip (November 15th 2019 at 12:00 PM UTC time is currently adopted, but note that if this time is in the past, your current time 
will be used instead).  Additionally, a vehicle's city and highway gas mileage rate are input (where the standard rates for a 2-axle car 
are currently used).  Adjust all of these rates as necessary to better represent your vehicle.  Lastly, a fuel cost rate of $3/gallon is 
currently set in this program, but do not change this if you will take the output API json file and use it with either toll_json_open.py 
or toll_analyzer.py (see below).  Those programs assume an input rate of $3/gallon but allow you to adjust it accordingly within their 
code.  This allows you to freely adjust the input fuel costs in your analyses with the need to rerun the Toll Guru API.

city_set.py: This is a simple program that is called by both toll_master_public.py and toll_analyzer.py.  It takes the csv file passed to
it (typically being top35.csv, which gives the 35 most populous cities in the United States) and reads in the start and end cities and
passes back their "city, state" strings that are both what necessary input parameters for the Toll Guru API and are also the basis of the
written/or called json filename output by the API or read into the main analysis programs.

toll_master_public.py:  This is the main program ran to call the Toll Guru API and outputs the appropriate json files that the 
toll_analyzer.py or toll_json_open.py programs analyze.  Its public name means that it calls toll_guru_api_series_public.py, which does
not include a Toll Guru API-Key.  It reads in the cities from top35.csv and generates all possible city to city routes.  Here the starting
city is always the more populous city.  For our cost analysis purposes, the route direction does not meaningfully change the calculation. 
For increased efficiency and limiting unnecessary calls to the API, the program checks if the corresponding output json file already 
exists for an input city to city route.  If it is exists, it continues to the next; if it does not, it calls the Toll Guru API and outputs
the result to a new json file.  Next, this program calls json_delete_empty.py (see below) to check whether an error occured in the output 
json file or if the API call limit was reached and an "empty" json files was returned, if yes, the just created json file is deleted and 
the program is exited.  Thi is because if an error or limit is reached, subsequent calls will most likely result in similarly bad json 
files.

json_delete_empty.py: This program can be called to check the status of an input json file, or a list of json files (see above).  
Otherwise, if this program is called directly, it reads in the list of all json files in the current directory and deletes all empty files
in the directory.

toll_analyzer.py: This is one of the main programs of the file and analyzes all city to city json files available in the current directory
and outputs 3 csv files (tollcalcdata.csv, onlyoneroute.csv, fastest_polylines.csv).  This file checks for various errors in the json 
files (see data cleaning discussion below), and performs the route efficiency calculations based on input maintenance costs 
(maintenance+depreciation per mile) and fuel $/gallon.  Additional, a single time cost $/hour is not input, but an upper limit is given 
(default is $30/hour) and the program calculates a result of the best route for the range of $0/hour to the input upper limit in 
increments of $1/hour, and this range is calculated for when the driver has toll tags and when the driver is paying with cash/license 
Plate.  Additionally, statistics are calculated for each individual route for how frequently the fastest route is the best, the cheapest 
route is the best, and neither route are the best (within the input $/hour range for both tags/no tags).  Next, the minimum $/hour rate 
for tags/no tags are calculated for where the fastest (typically most tolls) route is the best/cheapest route factoring in all costs. 
(This is not affected by the upper limit rate discussed above and consideres possible $/hour rates up to $2500/hour.)  If cash/license 
payment is not available for the fastest route, a NaN is returned.

onlyoneroute.csv: For the city to city routes where the fastest route is always the best route (under all hourly rates), these routes 
typically are toll free and only one "best" route is found, these city to city routes are output to onlyoneroute.csv, are checked for 
whether or not they have tolls and whether and if they do, it is checked whether or not cash/license payment is available.  Lastly, the 
total fuel costs, the google maps url of the route, and the encoded polyline of the route are output.

tollcalcdata.csv: For all other city to city routes, the best route for all $/hour tag/no tag combinations are output to tollcalcdata.csv.
Additionally, all the statistics discussed above are output, and the detailed route information for all available routes are given, 
including fuel costs, how much more the route costs in terms of maintenance, fuel, and tolls (with tags) relative to the cheapest route 
(excluding the additional, but variable, time costs).  Therefore, the cheapest route (excluding additional time costs) will always be 
given a 0.  Next, the time that the trip takes relative to the fastest route is given in minutes.  Therefore, the fastest route will 
always be given a 0.  Additionally, each routes google maps url is given for visual reference.  Lastly, because cash/license payments are
not always available, in some cases NaNs are returned for the "best route" where no routes are available with cash/license payment.  
Additionally, for the calculation of the hourly rate where the fastest route becomes the best route

fastest_polylines.csv: Additionally, because it is of interests to check the viability of the fastest route, more detailed information 
about the fastest route (again, if it's not also the cheapest route) is output to fastest_polylines.csv.  The minimum hourly rate where 
the fastest route is the best route for tags, and without tags (NaNs are given when the fastest route does not have cash/license payment 
available).  Lastly, the encoded polylines are included for the fastest route for subsequent display.

toll_json_open.py: This is a more specialized route analysis program, which only analyzes one city to city json file.  Like with 
toll_analyzer.py, an input maintenance+depreciation cost/mile and a fuel cost rate/gallon are given, but also now a specific user $/hour
rate and whether or not the driver has a tag or will pay with cash/license payment is input.  All output is put directly to screen, 
printing the city to city json file name.  Then which Route number is the best route at the given input setting is displayed.  Then 
individual costs are broken down and displayed:  1) Extra Toll Costs (how expensive are tolls relative to the cheapest toll route).  2) 
Extra Time Costs (how expensive, based on input hourly rate, is the additional time use relative to the fastest route).  3) Total Toll 
Costs (total toll cost of best route).  4) Total Travel Costs (total travel costs including all considered costs), 5) Total Travel Time 
(total time driving for the best route).  Lastly, the user is asked if they want to display in google maps the selected best route in the 
analysis.

toll_plots.py: This program takes the output tollcalcdata.csv and makes 3 informative plots of the data presented.  The top plots gives 
the cumulative distribution (for tags payment and for cash payment) of the minimum $/hour rate where the fastest route becomes the 
cheapest route.  For Routes without cash/license payment, they are included at the max of $2501/hour rate and included in the 
distribution, which allows its cumulative distribution be on the same scale as that of the tags payment distribution.  This analysis of 
the "fastest route" is of importance because it typically is the route that GPS systems automatically give you, regardless of additional 
costs of tolls, gas, and maintenance, etc.  The middle plot is a traditional histogram of the minimum $/hour rate where the fastest route
becomes the cheapest route, showing the full range up to two routes at roughly $2100/hour.  The lower panel shows the statistics for every
hourly rate across all input city to city json files, giving the percentage routes at this rate where the fastest route is best, the 
cheapest route is best, and neither route is best.  Neither being instances of where some, but not all, tolls are avoided, giving a route 
of intermediate toll+fuel costs and intermediate travel time.

route_maps.py: This produces quickly generated maps of all of the fastest city to city routes, giving their minimum $/hour rate where the
fastest route becomes the best/cheapest route.  (NOTE: Before running, see below for route_maps_polylinemerge.py, which produces more 
informative maps of much smaller file size, but that program takes much more time to run.)  The US maps are broken into tag rates and 
cash/license rates, and further broken down into cheap routes (< $15/hour), mid routes ($15 to $100/hour), and expensive routes 
($100+/hour).  Polyline routes are shown on a US map for each fastest route, and starting and ending markers are given for each city.  The
route lines are color coded based on cost and have their own popup when clicked, which shows both which city to city route this line 
represents and what its minimum hourly rate is.  The markers for each city are merged, allowing for a fully detailed popup to be displayed
when the marker is selected, which shows all routes where this city is either the starting point or ending point, also giving that route's
corresponding minimum hourly rate.  Additionally, a separate map is created that shows all of the fastest routes where cash/license 
payment is unavailable.  Another map is created for the routes from onlyoneroute.csv, where the fastest route is also the cheapest route,
for all hourly rates.  The routes on this separate map are color coded based on whether or not these routes are toll free or have tolls, 
and if they do, whether or not these tolls allow cash/license payment.  

These maps are output as individual html files.  The generation of these are relatively efficient (on my computer I can generate all maps
at a rate of ~500 routes in ~1 minute), but they have limitations and have very large file sizes because all polyline routes are plotted 
on the same map and commonly overlap on top of each other.  Due to a majority of these routes following along common interstate systems,
only the last polyline plotted is visible, and when this polyline is selected, only the last route's popup information is displayed. 
Therefore, the underlying information adds to the file size, but it does not add to the displayed information available to the user.  To 
overcome this, see route_maps_polylinemerge.py.  

route_maps_polylinemerge.py: To overcome the limitations discussed above, an algorithm was added to the foundation of route_maps.py that 
merges polyline information, which allows a single polyline route to be plotted at a given location instead of many overlapping polylines. 
Addtionally, the displayed popup information is merged at each point, providing the route information of all routes that pass through the 
selected polyline location.  This produces much smaller map file sizes while also giving the ability to display all route information.
These output maps are superior in all aspects, but this algorithm is slower (~500 routes in ~7 minutes), hence, I have also included 
route_maps.py for a more efficient example.


Concepts and Statistics Addressed/Applied in This Project and Ideas for Future Expansion:

Data Cleaning:
Handling incomplete data:  The Toll Guru API outputs None for the cash or LicensePlate toll costs when either of the toll payment types is
unavailable.  toll_analyzer.py handles these for calculations when the user selects cash/license payments by changing these Nones to a 
$10,000 costs.  This allows these numbers to be automatically put into the routes calculations but effectively show these are unavailable
to cash/licensplate drivers.  When calculating general statistics, this essentially makes these routes are quantitatively unreasonable in
the comparison to routes without tolls or with cash/license available tolls.  In the rare case where no available routes are without tolls
or have cash/license payment available, the programs check for these and makes note of this in the output.  Lastly, a dollar value for
these tolls without cash/license payment allows the cumulative distribution of minimum toll costs (toll_plots.py) to include them in the 
set and keep the sample's scale to be consistent with the tag-based toll rates.

Handling Data Errors: For a small number of the output Toll Guru API files, the comparative cost values were shown to be in error (or just
None values were given).  Therefore, toll_analyzer.py and toll_json_open.py ignores these relative cost values and calculates them from 
the raw cost information.  Consistency checks show that these results are nearly always consistent with the comparative cost output 
directly, but catching the small number of errors is important and they would occasionally crash the program.  Furthermore, breaking each
of the individual costs down allows for straightforward scaling of gas costs, gas mileage rates, and the addition of maintenance costs.  
Many, but not all, of these additions are available directly in the API, but it is more efficient to perform these various calculations 
from the raw data rather than producing a new and independent json file API output.

Statistical Analysis: The large data sets taken from the toll guru API, where we have tracked all available routes between the ~30 largest 
cities in the US, are analyzed and the cost comparisons, which factor in gas, car maintenance costs per mile, toll costs, and the personal
costs of your time, are calculated for all available routes.  The primary variable in this analysis is the personal cost of time, at each
time from $1/hour to $30/hour, what is the most cost-effective available route.  This analysis shows that when time is of little value to
the driver/passengers, the route with minimal tolls+gas costs is the ideal route.  This route is typically the "avoid tolls" route given
by most GPS/routing programs.  However, as the cost of time becomes more important on your trip, the fastest route with typically more
tolls and more miles at higher speeds becomes the ideal route.  

For drivers with toll tags, this transition of where the fastest route is typically best occurs at $22.50/hour.  For drivers with cash 
toll payments, which typically are more expensive, this transition occurs at $30.50/hour.  It is important to note that independent of 
adopted hourly rates, 10 to 15% of the time, neither the fastest nor the cheapest route is best, and some intermediate route that avoid 
some but not all tolls is ideal.  Such routes are not typically an automatic option in GPS programs where all or none of the tolls are 
typically given as the option.  The median hourly payment rate of a US worker is ~$22.50/hour.  Therefore, adopting this as a typical cost
trade off for an American driver, the costs of taking the toll routes with tags between cities is half of the time to your benefit and 
half of the time to your disadvantage.  However, this is not the case when you are paying with cash.  Additionally, this is only 
illustrative because the costs of taking tolls between different city to city routes vary significantly.

Extremely High Cost Routes: Many of the fastest routes between cities come at a significant cost relative to marginally slower routes that
are overall shorter distance drives and in many cases do not include tolls.  For example, when paying tolls with cash, approximately 50% 
of the fastest city to city routes either cost the driver a rate at least $50/hr vs the time saved, or cash payment options are simply not
available.  With tags for toll payments, roughly 50% of the fastest city to city routes cost the driver at least $23.50/hr.  In the most 
extreme of cases, however, the top 10% of tags-based fastest routes cost anywhere from $250 to an excessive $2102/hr.  Such extreme 
costroutes should be avoided by virtually all drivers.

Statistical Applications: For drivers where cost and efficiency are important, these results show that when time is of low cost to the 
driver (< $20/hour), they will generally want to avoid as many tolls as possible.  Additionally, when time is of high cost to the driver 
(> $30/hour), they will generally want to take the fastest route with more tolls.  However, these cases are highly variable for differing 
city to city routes and this information is most effective on a case by case basis for an individual driver.  Such a program integrated 
into a phone's GPS would truly incorporate the fact that time is money and consider the true costs of car travel.

Program Expansion:  Factoring in all costs of driving, including the time costs and maintenance+depreciation costs, allows for the total 
drive costs to be calculated for a given city to city trip (or for round trip).  For short notice trips, driving is usually ideal, 
especially for relatively short trips (e.g., < 8 hrs).  However, for travellers that are able to plan in advance, a travel program 
building on mine can also call additional travel APIs for flights, trains, and buses and estimate travel costs through these methods, 
including the cost of time through these travel methods, plus include costs of travelling/parking from your location to the appropriate 
airport/station.  Initial comparisons (using my program) of driving costs to flight costs from popular travel websites shows that in 
general, flights are almost always cheaper when travelling alone and prepared early and in advance, but for short trips of multiple 
people, the car costs are significantly better.  
