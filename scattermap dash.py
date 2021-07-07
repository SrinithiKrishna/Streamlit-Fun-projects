import plotly.graph_objs as go
#<density_scatter_mapbox>
@app.callback(
 Output('map-quakes', 'children'),
 [Input('past-occurrence', 'value'), Input('magnitude-range', 'value'), Input('map-type', 'value'), Input('area-list', 'value'), 
  Input('output-update', 'n_intervals')],
)
def visualize_quakes(past_occurrence, mag_value, map_type, specific_area, n_intervals):
   try:
      eqdf = GrabOccurrenceData(past_occurrence, mag_value)
      eqdf = eqdf[eqdf['place'].str.contains(str(specific_area.split(' - ')[0]))]
      zoom = 3
      radius = 15
      latitudes = eqdf['latitude'].to_list()
      longitudes = eqdf['longitude'].to_list()
      magnitudes = eqdf['mag'].to_list()
      mags = [float(i)*radius_multiplier['outer'] for i in magnitudes]
      mags_info = ['Magnitude : ' + str(m) for m in magnitudes]
      depths = eqdf['depth'].to_list()
      deps_info = ['Depth : ' + str(d) for d in depths]
      places = eqdf['place'].to_list()
      center_lat = eqdf[eqdf['mag'] == eqdf['mag'].max()['latitude'].to_list()[0]
      center_lon = eqdf[eqdf['mag'] == eqdf['mag'].max()]['longitude'].to_list()[0]

      if (map_type == 'Density Map'):
         map_trace = PlotDensityMap(latitudes, longitudes, magnitudes, radius, 'Electric')
         layout_map = LayoutDensity(600, 980, 'stamen-terrain', center_lat, center_lon, zoom)
         visualization = html.Div([
         dcc.Graph(
         id='density-map',
         figure={'data' : [map_trace], 'layout' : layout_map}
         ),
         ])
      return visualization
      if (map_type == 'Scatter Map'):
         quake_info = [places[i] + '<br>' + mags_info[i] + '<br>' + deps_info[i] for i in range(eqdf.shape[0])]
         map_trace = PlotScatterMap(latitudes, longitudes, mags, magnitudes, default_colorscale, quake_info)
         layout_map = LayoutScatter(600, 980, 'stamen-terrain', center_lat, center_lon, zoom)
         visualization = html.Div([
         dcc.Graph(
         id='scatter-map',
         figure={'data' : [map_trace], 'layout' : layout_map}
         ),
         ])
      return visualization
   except Exception as e:
      return html.Div([
   html.H6('Please select valid magnitude / region ...')
  ], style={'margin-top' : 150, 'margin-bottom' : 150, 'margin-left' : 200})
#</density_scatter_mapbox>\

if __name__ == '__main__':
 app.run_server(debug=True, dev_tools_props_check=False, dev_tools_ui=False)