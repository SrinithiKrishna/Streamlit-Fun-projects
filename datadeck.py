import streamlit as st
data = pd.DataFrame({
    'awesome cities' : ['Chicago', 'Minneapolis', 'Louisville', 'Topeka'],
    'lat' : [41.868171, 44.979840,  38.257972, 39.030575],
    'lon' : [-87.667458, -93.272474, -85.765187,  -95.702548]
})

# Adding code so we can have map default to the center of the data
midpoint = (np.average(data['lat']), np.average(data['lon']))

st.deck_gl_chart(
            viewport={
                'latitude': midpoint[0],
                'longitude':  midpoint[1],
                'zoom': 4
            },
            layers=[{
                'type': 'ScatterplotLayer',
                'data': data,
                'radiusScale': 250,
   'radiusMinPixels': 5,
                'getFillColor': [248, 24, 148],
            }]
        )