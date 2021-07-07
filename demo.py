import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from streamlit_folium import folium_static
import folium
from branca.element import Figure

st.title("Geovisualization")
st.sidebar.title("A step towards collabarative decision making")
st.markdown(" This interactive App is to analyze and visualize the Socio-economic status of Indian states")
st.sidebar.markdown(" It helps the top level Government higher authorities and decision makers to carry out the development plan")
selected_year=st.sidebar.selectbox('year',list(reversed(range(2010-2021))))
@st.cache
def main():
    menu=["Home","map","charts"]
    choice=st.sidebar.selectbox("Menu",menu)
    if choice=='Home':
        st.subheader("Home")
        df=pd.read_csv("cities_r2.csv")
        st.line_chart(df)
        m = folium.Map(location=[28.644800, 77.216721], zoom_start=16)
        fig=Figure(width=550,height=350)
        folium.TileLayer('Stamen Terrain').add_to(m)
        folium.TileLayer('Stamen Toner').add_to(m)
        folium.TileLayer('Stamen Water Color').add_to(m)
        folium.TileLayer('cartodbpositron').add_to(m)
        folium.TileLayer('cartodbdark_matter').add_to(m)
        folium.LayerControl().add_to(m)
# add marker
        tooltip = "India"
        folium.Marker([28.644800, 77.216721], popup="india rock", tooltip=tooltip).add_to(m)


# call to render Folium map in Streamlit
        folium_static(m)

    elif choice=='Maps':
        st.subheader("Maps")
        
    elif choice=='charts':
        st.subheader("charts")
    else:
        st.subheader("About us")


if __name__=='__main__':
    main()
        



