import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input,Output
import folium
import dash_table_experiments as dash_table

app = dash.Dash()

app.layout = html.Div([
    html.H1('My first app with folium map'),
    html.Iframe(id='map', srcDoc=open('testmap1.html', 'r').read(), width='100%', height='600'),
    html.Button(id='map-submit-button', n_clicks=0, children='Submit')
])


@app.callback(
    dash.dependencies.Output('map', 'srcDoc'),
    [dash.dependencies.Input('map-submit-button', 'n_clicks')])
def update_map(n_clicks):
    if n_clicks is None:
        return dash.no_update
    else:
        return open('testmap1.html', 'r').read()

if __name__ == '__main__':
    app.run_server(debug=True)