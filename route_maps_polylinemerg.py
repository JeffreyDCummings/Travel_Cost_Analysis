""" Take the output csv files from toll_scraper.py and produce several route maps with the
key information shown with color coded polylines, detailed popups, and multi-level start
and stop markers (to handle overlapping routes). """
import os
import pandas as pd
import polyline
import folium
import branca.colormap as cm

MAX_CHEAP = 15
MAX_MID = 100
MAX_EXPENSIVE = 900
MAP_CENTER = [38.4282, -97.5795]
INITIAL_ZOOM = 5

def define_marker_popups(latlon_dict_in, row_in, additional_text):
    """ Create a dictionary with city coordinates as the key and append route popup
    information as the dictionary value each time the coordinates (city) reappear. """
    start_tuple, stop_tuple = tuple(row_in["latlon"][0]), tuple(row_in["latlon"][-1])
    if start_tuple in latlon_dict_in.keys():
        latlon_dict_in[start_tuple] = latlon_dict_in[start_tuple]+"<br>"+\
         str(row_in["Cities"])+additional_text
    else:
        latlon_dict_in[start_tuple] = str(row_in["Cities"])+additional_text
    if stop_tuple in latlon_dict_in.keys():
        latlon_dict_in[stop_tuple] = latlon_dict_in[stop_tuple]+"<br>"+\
         str(row_in["Cities"])+additional_text
    else:
        latlon_dict_in[stop_tuple] = str(row_in["Cities"])+additional_text
    return latlon_dict_in

def save_html_cor(mapit, html_file):
    """Write generated map to an html file, and fix its marker popup formatting. """
    mapit.save(html_file)
    with open(html_file, "rt") as filein:
        with open("temp.html", "wt") as fileout:
            for line in filein:
                fileout.write(line.replace('{"maxWidth": "100%"}', '{"maxWidth": 500}'))
    os.rename("temp.html", html_file)

def toll_map_rates(row_in, popup_message, latlon_dict_in, color_input, route_dict_in):
    """ Define route popup, draw route polyline, and add its start and stop points to the
     marker dictionary. """
    for point in range(len(row_in["latlon"])-1):
        latlon_route = tuple(((row_in["latlon"][point]), (row_in["latlon"][point+1])))
        if latlon_route in route_dict_in.keys():
            route_dict_in[latlon_route][0] = route_dict_in[latlon_route][0]+"<br>"+\
            str(row_in["Cities"])+popup_message
        else:
            route_dict_in[latlon_route] = [str(row_in["Cities"])+popup_message, color_input]
    latlon_dict_in = define_marker_popups(latlon_dict_in, row_in, popup_message)

def add_marker_popups(latlon_dict_in, mapit):
    """ Iterate through the dictionary of cities (coordinates) and add their markers and
    cooresponding values as the popup. """
    for point in latlon_dict_in:
        folium.Marker(point, popup=latlon_dict_in[point]).add_to(mapit)

def color_map_set():
    """ Set the different ranges of hourly rates, generate color_maps that span them, and
     add a caption to the color_map legend. """
    colormapcheap = cm.LinearColormap(colors=['purple', 'blue', 'lightblue', 'green'],\
     vmin=1, vmax=MAX_CHEAP)
    colormapmid = cm.LinearColormap(colors=['green', 'lightgreen', 'yellow', 'orange'],\
     vmin=MAX_CHEAP, vmax=MAX_MID)
    colormapexpensive = cm.LinearColormap(colors=['orange', 'salmon', 'red', 'darkred'],\
     vmin=MAX_MID, vmax=MAX_EXPENSIVE)
    colormapcheap.caption = 'The time you save costs you this hourly rate ($/hr)'
    colormapmid.caption = 'The time you save costs you this hourly rate ($/hr)'
    colormapexpensive.caption = 'The time you save costs you this hourly rate ($/hr)'
    return [colormapcheap, colormapmid, colormapexpensive]

def initialize_map(color_map=None):
    """ Create the map object and define its center and zoom, and add a color_map if
     it is defined """
    mapit = None
    mapit = folium.Map(location=MAP_CENTER, zoom_start=INITIAL_ZOOM, control_scale=True)
    if color_map is not None:
        mapit.add_child(color_map)
    return mapit

def polyline_dataframe_setup(toll_polyline_csv, onlyoneroute_csv):
    """Read in csv files output by toll_scraper, which includes key toll calculations
     and encoded polylines.  Decode polylines, and from the toll dataframe, create
     two dataframes sorted by cash or tags rates and return.  Similarly, return the only
     one viable route dataframe. """
    polylinesdf = pd.read_csv(toll_polyline_csv, delimiter=",")
    polylinesdf["latlon"] = polylinesdf["Polyline"].apply(polyline.decode)
    polylinestagsdf = polylinesdf.sort_values("Min Rate Tags", ascending=False)
    polylinescashdf = polylinesdf.sort_values("Min Rate Cash", ascending=False)
    polylinesoneroutedf = pd.read_csv(onlyoneroute_csv, delimiter=",")
    polylinesoneroutedf["latlon"] = polylinesoneroutedf["Polyline"].apply(polyline.decode)
    return polylinestagsdf, polylinescashdf, polylinesoneroutedf

def dictionary_merge(map_in, dict_in):
    """Merge all polylines and popup info for a map, providing more visible information
    yet smaller map file sizes."""
    dict_out = {}
    while len(dict_in) > 1:
        keys_list = list(dict_in.keys())
        keys_match_list = [keys_list[0]]
        for step in dict_in:
            if dict_in[keys_list[0]][0] == dict_in[step][0] and step != keys_list[0]:
                keys_match_list.append(step)
        keys_merged_list = [list(keys_match_list[0])]
        for key in range(len(keys_match_list)-1):
            if keys_merged_list[-1][-1] == keys_match_list[key+1][0]:
                keys_merged_list[-1].append(keys_match_list[key+1][1])
            else:
                keys_merged_list.append(list(keys_match_list[key+1]))
        for keys_merged in keys_merged_list:
            dict_out.update({tuple(keys_merged): dict_in[keys_list[0]]})
        for key in keys_match_list:
            del dict_in[key]
    if len(dict_in) == 1:
        dict_out.update(dict_in)
    for latlon_route, route_info in reversed(list(dict_out.items())):
        folium.PolyLine(latlon_route, popup=route_info[0],\
         color=route_info[1], weight=3.5).add_to(map_in)

def build_tags_maps(polylinestagsdf, map_dict):
    """Iterate over the dataframe of polylines sorted by tags, and add the information
     to the Cheap, Mid, and Expensive tags map. """
    for _, row in polylinestagsdf.iterrows():
        if row["Min Rate Tags"] < MAX_CHEAP:
            toll_map_rates(row, " ($"+str(int(row["Min Rate Tags"]))+"/hr)",\
             map_dict["tags_cheap"][2], map_dict["tags_cheap"][1](row["Min Rate Tags"]),\
             map_dict["tags_cheap"][4])
        elif row["Min Rate Tags"] < MAX_MID:
            toll_map_rates(row, " ($"+str(int(row["Min Rate Tags"]))+"/hr)",\
             map_dict["tags_mid"][2], map_dict["tags_mid"][1](row["Min Rate Tags"]),\
             map_dict["tags_mid"][4])
        else:
            toll_map_rates(row, " ($"+str(int(row["Min Rate Tags"]))+"/hr)",\
             map_dict["tags_exp"][2], map_dict["tags_exp"][1](row["Min Rate Tags"]),\
             map_dict["tags_exp"][4])

def build_one_route_map(polylinesoneroutedf, map_dict):
    """Iterate over the dataframe of polylines of toll free/toll only routes, and add
     the information to one route map. """
    for _, row in polylinesoneroutedf.iterrows():
        if row["Has Tolls"]:
            if row["Cash/License Plates Available"]:
                toll_map_rates(row, " The Toll Route Is Best For All Rates",\
                 map_dict["one_route"][2], "blue", map_dict["one_route"][4])
            else:
                toll_map_rates(row, " The Toll Route Is Best For All Rates But Requires Tags",\
                 map_dict["one_route"][2], "red", map_dict["one_route"][4])
        else:
            toll_map_rates(row, " Is Toll Free", map_dict["one_route"][2],\
             "green", map_dict["one_route"][4])

def build_cash_maps(polylinescashdf, map_dict):
    """Iterate over the dataframe of polylines sorted by cash, and add the information
     to the Cheap, Mid, and Expensive cash maps and cash/license plate unavilable map. """
    for _, row in polylinescashdf.iterrows():
        if pd.isnull(row["Min Rate Cash"]):
            toll_map_rates(row, " Cash/License Plate Tolls Unavailable.",\
             map_dict["cash_nan"][2], 'black', map_dict["cash_nan"][4])
        else:
            if row["Min Rate Cash"] < MAX_CHEAP:
                toll_map_rates(row, " ($"+str(int(row["Min Rate Cash"]))+"/hr)",\
                 map_dict["cash_cheap"][2], map_dict["cash_cheap"][1](row["Min Rate Cash"]),\
                 map_dict["cash_cheap"][4])
            elif row["Min Rate Cash"] < MAX_MID:
                toll_map_rates(row, " ($"+str(int(row["Min Rate Cash"]))+"/hr)",\
                 map_dict["cash_mid"][2], map_dict["cash_mid"][1](row["Min Rate Cash"]),\
                 map_dict["cash_mid"][4])
            else:
                toll_map_rates(row, " ($"+str(int(row["Min Rate Cash"]))+"/hr)",\
                 map_dict["cash_exp"][2], map_dict["cash_exp"][1](row["Min Rate Cash"]),\
                 map_dict["cash_exp"][4])

def add_merged_polylines_series(map_dict):
    """A series of calls to add the merged polylines to every map objects."""
    for map_in in map_dict:
        dictionary_merge(map_dict[map_in][0], map_dict[map_in][4])

def add_marker_series(map_dict):
    """A series of calls to add the markers to every map objects."""
    for map_in in map_dict:
        add_marker_popups(map_dict[map_in][2], map_dict[map_in][0])

def html_series_save(map_dict):
    """A series of calls to save the 8 map objects to an html map."""
    for map_in in map_dict:
        save_html_cor(map_dict[map_in][0], map_dict[map_in][3])

def main():
    """Call functions to setup color maps, generate processed dataframes, initialize maps,
     then map polylines, markers, and route information to maps, and then output to html. """

    colormap_tags = color_map_set()
    colormap_cash = color_map_set()

    map_dict =\
    {"tags_cheap":[initialize_map(color_map=colormap_tags[0]), colormap_tags[0], {}, "maptagscheap.html", {}],\
     "tags_mid":[initialize_map(color_map=colormap_tags[1]), colormap_tags[1], {}, "maptagsmid.html", {}],\
     "tags_exp":[initialize_map(color_map=colormap_tags[2]), colormap_tags[2], {}, "maptagsexpensive.html", {}],\
     "cash_cheap":[initialize_map(color_map=colormap_cash[0]), colormap_cash[0], {}, "mapcashcheap.html", {}],\
     "cash_mid":[initialize_map(color_map=colormap_cash[1]), colormap_cash[1], {}, "mapcashmid.html", {}],\
     "cash_exp":[initialize_map(color_map=colormap_cash[2]), colormap_cash[2], {}, "mapcashexpensive.html", {}],\
     "one_route":[initialize_map(), None, {}, "maponeroute.html", {}],\
     "cash_nan":[initialize_map(), None, {}, "mapcashunavailable.html", {}]}

    polylinestagsdf, polylinescashdf, polylinesoneroutedf =\
     polyline_dataframe_setup("fastest_polylines.csv", "onlyoneroute.csv")

    build_tags_maps(polylinestagsdf, map_dict)
    build_one_route_map(polylinesoneroutedf, map_dict)
    build_cash_maps(polylinescashdf, map_dict)
    add_merged_polylines_series(map_dict)
    add_marker_series(map_dict)

    html_series_save(map_dict)

if __name__ == "__main__":
    main()
