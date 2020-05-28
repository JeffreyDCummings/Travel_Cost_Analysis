import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

# plotly plot embed urls
HTMLS = {
    'Cash - Cheap Routes (The cost of the time you are saving: < $15/hr)': open('mapcashcheap.html', 'r').read(),
    'Cash - Mid Routes ($15/hr to $100/hr)': open('mapcashmid.html', 'r').read(),
    'Cash - Expensive Routes (> $100/hr)': open('mapcashexpensive.html', 'r').read(),
    'Tags - Cheap Routes (< $15/hr)': open('maptagscheap.html', 'r').read(),
    'Tags - Mid Routes ($15/hr to $100/hr)': open('maptagsmid.html', 'r').read(),
    'Tags - Expensive Routes (> $100/hr)': open('maptagsexpensive.html', 'r').read(),
    'Only One Viable Route': open('maponeroute.html', 'r').read(),
    'No cash/license-plate tolls available': open('mapcashunavailable.html', 'r').read(),
    'Individual Toll Rate Info': open('tollinfo.html', 'r').read()
}

keys = list(HTMLS.keys())
vals = list(HTMLS.values())

# initialize Dash object
app = dash.Dash()

# define dropdown whose options are embed urls
dd = dcc.Dropdown(id='dropdown',\
 options=[{'label': k, 'value': v} for k, v in zip(keys, vals)], placeholder=keys[0])

# embedded plot element whose `src` parameter will
# be populated and updated with dropdown values
plot = html.Iframe(id='plot', style={'border': 'none', 'width': '100%', 'height': 700},\
 srcDoc=vals[0])

# set div containing dropdown and embedded plot
app.layout = html.Div(children=[dd, plot])

# update `src` parameter on dropdown select action
@app.callback(Output(component_id='plot', component_property='srcDoc'),\
 [Input(component_id='dropdown', component_property='value')])

def update_plot_srcdoc(input_value):
    return input_value

if __name__ == '__main__':
    app.run_server(port=8000, host='127.0.0.1')
