""" Plot the key information about the statistics of the minimum hourly rates where
 a given route becomes the best route.  Additionally, plot how frequently the fastest,
 cheapest, or neither routes are best for a given hourly rate up to HOURLY_RATE_MAX. """
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

HOURLY_RATE_MAX = 30

def list_initialize():
    """ Initialize to zero two list of lists for all scenarios. """
    fastest_best_tags = [0 for x in range(HOURLY_RATE_MAX+1)]
    neither_best_tags = [0 for x in range(HOURLY_RATE_MAX+1)]
    cheapest_best_tags = [0 for x in range(HOURLY_RATE_MAX+1)]
    fastest_best_cash = [0 for x in range(HOURLY_RATE_MAX+1)]
    neither_best_cash = [0 for x in range(HOURLY_RATE_MAX+1)]
    cheapest_best_cash = [0 for x in range(HOURLY_RATE_MAX+1)]
    return [fastest_best_tags, neither_best_tags, cheapest_best_tags], \
     [fastest_best_cash, neither_best_cash, cheapest_best_cash]

def tags_rate_count(route_info, hourly_rate, best_tags, size_tags):
    """ Count the percentage of times with tags where the best route is the fastest [0],
     cheapest [2], or neither [1]. """
    best_route_tag = route_info["$"+str(hourly_rate)+"/hr with tags"]
    if route_info.loc["Fastest "+best_route_tag.split()[1]] == 0:
        best_tags[0][hourly_rate] += 1/size_tags
    elif route_info.loc["Cheapest "+best_route_tag.split()[1]] == 0:
        best_tags[2][hourly_rate] += 1/size_tags
    else:
        best_tags[1][hourly_rate] += 1/size_tags
    return best_tags

def cash_rate_count(route_info, hourly_rate, best_cash, size_cash):
    """ Count the number of times with cash where the best route is the fastest [0],
     cheapest [2], or neither [1].  Additionally, check for routes with unavailable
     cash payment and decrement size_cash."""
    best_route_cash = route_info["$"+str(hourly_rate)+"/hr with cash"]
    if isinstance(best_route_cash, str):
        if route_info.loc["Fastest "+best_route_cash.split()[1]] == 0:
            best_cash[0][hourly_rate] += 1
        elif route_info.loc["Cheapest "+best_route_cash.split()[1]] == 0:
            best_cash[2][hourly_rate] += 1
        else:
            best_cash[1][hourly_rate] += 1
    elif hourly_rate == 0:
        size_cash -= 1
    return best_cash, size_cash

def percentage_calculations(tolldf):
    """ Iterate through all available route sets and hourly rates, counting percentages
     of where the fast, cheap, or neither are the best route. """
    size_tags, size_cash = len(tolldf), len(tolldf)
    best_tags, best_cash = list_initialize()
    for _, route_info in tolldf.iterrows():
        for hourly_rate in range(HOURLY_RATE_MAX+1):
            best_tags = tags_rate_count(route_info, hourly_rate, best_tags, size_tags)
            best_cash, size_cash = cash_rate_count(route_info, hourly_rate, best_cash, size_cash)
    for hourly_rate in range(HOURLY_RATE_MAX+1):
        for route_type in range(3):
            best_cash[route_type][hourly_rate] /= size_cash
    return best_tags, best_cash

def percentage_plot(best_tags, best_cash):
    """ Plot the percentages of either fast, cheap, or neither being best as a line plot."""
    plot_legend = plt.subplot(313)
    plt.xlabel("$/Hour Rate")
    plt.ylabel("Percentage")
    plt.plot(range(HOURLY_RATE_MAX+1), best_tags[2], color='blue', label='Cheap Best Tags')
    plt.plot(range(HOURLY_RATE_MAX+1), best_tags[0], color='black', label='Fast Best Tags')
    plt.plot(range(HOURLY_RATE_MAX+1), best_tags[1], color='red', label='Neither Best Tags')
    plt.plot(range(HOURLY_RATE_MAX+1), best_cash[2], '--', color='blue', label='Cheap Best Cash')
    plt.plot(range(HOURLY_RATE_MAX+1), best_cash[0], '--', color='black', label='Fast Best Cash')
    plt.plot(range(HOURLY_RATE_MAX+1), best_cash[1], '--', color='red', label='Neither Best Cash')
    plot_legend.legend()

def min_rate_cumulative(tolldf):
    """ Plot the cumulative distribution of the minimum hourly rates. """
    plt.subplot(311)
    plt.xlabel("Minimum $/Hour Rate Where The Fastest Route Becomes the Cheapest Route")
    plt.ylabel("Percentage")
    plt.xlim(-5, 400)
    bins = np.linspace(0, 2501, 1200)
    plt.hist(tolldf["Min Rate Tags"], bins, alpha=0.5, color="red",\
     label="Minimum Hrly rate with tags", histtype='step', linewidth=2, cumulative=True, density=1)
    plt.hist(tolldf["Min Rate Cash"], bins, alpha=0.5, color="blue",\
     label="Minimum Hrly rate with cash", histtype='step', linewidth=2, cumulative=True, density=1)
    plt.legend(prop={'size': 10}, loc=4)

def min_rate_hist(tolldf):
    """ Plot the histogram of the minimum hourly rates. """
    plt.subplot(312)
    plt.xlabel("Minimum $/Hour Rate Where The Fastest Route Becomes the Cheapest Route")
    plt.ylabel("Number")
    plt.xlim(-5, 2300)
    plt.ylim(0, 13)
    bins = np.linspace(0, 2500, 51)
    plt.hist(tolldf["Min Rate Tags"], bins, alpha=0.5, color="red",\
     label="Minimum Hrly rate with tags", histtype='step', linewidth=2)
    plt.hist(tolldf["Min Rate Cash"], bins, alpha=0.5, color="blue",\
     label="Minimum Hrly rate with cash", histtype='step', linewidth=2)
    plt.legend(prop={'size': 10})

def main():
    """ Open the csv data file, replace NaNs, and plot the key data. """

    tolldf = pd.read_csv("tollcalcdata.csv", delimiter=",")
    tolldf.fillna(value=2501, inplace=True)

    plt.figure(0, figsize=(12, 10))

    best_tags, best_cash = percentage_calculations(tolldf)

    percentage_plot(best_tags, best_cash)
    min_rate_cumulative(tolldf)
    min_rate_hist(tolldf)
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()
