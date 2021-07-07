import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
from PIL import Image
from streamlit_folium import folium_static
import folium
from branca.element import Figure
import streamlit.components.v1 as components  # Import Streamlit
from pathlib import Path
import lux


# Render the h1 block, contained in a frame of size 200x200.

def main():
    menu=["Home","Maps","charts"]
    choice=st.sidebar.selectbox("Menu",menu)
    if choice=='Home':
        components.html("<html><p><h1>National Remote Sensing center(ISRO)</h1></p></html>", width=1200, height=100)
        st.success("**Geovisualization Portal for Literacy,Crime,Unemployement effect in GDP**")
        image = Image.open('globe.jpg')
        st.image(image,use_column_width=True)
        st.title("Geovisualization")
        st.sidebar.title("A step towards collabarative decision making")
        st.markdown(" This interactive App is to analyze and visualize the Socio-economic status of Indian states")
        st.sidebar.markdown(" It helps the top level Government higher authorities and decision makers to carry out the development plan")
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            df = pd.read_excel(uploaded_file)
            st.write(df)
  # Add some matplotlib code !
            fig, ax = plt.subplots()
            df.hist(
                bins=8,
                column="Per Capita NSDP",
                grid=False,
                figsize=(8, 8),
                color="#86bf91",
                zorder=2,
                rwidth=0.9,
                ax=ax,
            )
            st.write(fig)
    elif choice=='Maps':
        st.title('Map Visualization')    
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

  
    elif choice=='charts':
        st.title('Generate EDA')
        def app():
            st.write(' Data Exploratory Data Analysis ')
            df = pd.read_csv("C:\\Users\\srini\\OneDrive\\Desktop\\Mission Project ISRO\\statewise dataset\\cities_r2.csv")
            export_file = 'visualizations.html'
            df.save_as_html(export_file)
            txt = Path(export_file).read_text()
            components.html(txt, width=800, height=350)
        app()
    else:
        st.subheader("About us")


if __name__=='__main__':
    main()
        



