#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


path1="C:\\Users\\srini\\OneDrive\\Desktop\\Mission Project ISRO\\statewise dataset\\cities_r2.csv"
city_data=pd.read_csv(path1)
city_data.head()


# In[3]:


city_data.describe()


# In[4]:


group_cit=city_data.iloc[:,2:].groupby("state_name")
cities=group_cit.mean()
cities['state_name']=cities.index
cols = cities.columns.tolist()
cols = cols[-1:] + cols[:-1]
cities=cities[cols]
cities.head()


# In[5]:


cities.describe()


# In[6]:


path2="C:\\Users\\srini\\OneDrive\\Desktop\\Mission Project ISRO\\statewise dataset\\UnEmployment.csv"
unemployment_data=pd.read_csv(path2)
unemployment_data.head()


# In[7]:


unemployment_data.describe()

# In[8]:


path3="C:\\Users\\srini\\OneDrive\\Desktop\\Mission Project ISRO\\statewise dataset\\Sucide_Rate.csv"
suicide_data=pd.read_csv(path3)
suicide_data.head()


# In[9]:


suicide_data.describe()


# In[10]:


path4="C:\\Users\\srini\\OneDrive\\Desktop\\Mission Project ISRO\\statewise dataset\\Crime.csv"
crime_data=pd.read_csv(path4)
crime_data.head()


# In[11]:


crime_data.describe()


# In[12]:


df=pd.merge(unemployment_data,crime_data,on="State",how='outer')


# In[13]:



df1=pd.merge(df,suicide_data,on="State",how='outer')


# In[14]:


df1.head()


# In[15]:


df1.describe()


# In[16]:


cols=df1.columns
cols


# In[17]:


df1.columns
df1.set_index("State",inplace=True)
df1.columns


# In[18]:


from sklearn.preprocessing import StandardScaler


from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import numpy as np


# In[19]:


df1_standardized = StandardScaler().fit_transform(df1)
#df1_standardized


# In[20]:


pca = PCA(n_components=8)
final = pca.fit(df1_standardized)


# In[21]:


var=final.explained_variance_ratio_
var


# In[22]:


tot=np.cumsum((final.explained_variance_ratio_))
tot


# In[23]:


Absfcomp=abs( final.components_ )
Absfcomp


# In[24]:


final_comp=pd.DataFrame(final.components_)
final_comp


# In[25]:


n_pcs= pca.components_.shape[0]
most_important = [np.abs(final.components_[i]).argmax() for i in range(n_pcs)]
most_important

initial_feature_names = ['0','1','2','3','4','5','6','7']

most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]

dic = {'PC{}'.format(i): most_important_names[i] for i in range(n_pcs)}

imp_df= pd.DataFrame(dic.items())
print(imp_df)


# In[26]:


cluster=KMeans(n_clusters=5,n_init = 12,random_state=0)
cluster.fit(df1_standardized)


# In[27]:


y_kmeans = cluster.fit_predict(df1_standardized)
y_kmeans


# In[28]:


df1["Clusters"]=cluster.fit_predict(df1_standardized)


# In[29]:


cols=df1.columns
cols


# In[30]:


pca1=PCA(n_components=3)
df1["X"]=pca1.fit_transform(df1[cols])[:,0]
df1["Y"]=pca1.fit_transform(df1[cols])[:,1]
df1["Z"]=pca1.fit_transform(df1[cols])[:,2]
df1=df1.reset_index()





# In[31]:


new_clusterData=df1[["State","Clusters","X","Y","Z"]]


# In[32]:


new_clusterData.tail()


# In[33]:


# Imports
from sklearn import mixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# In[34]:


reduced_data=df1[["X","Y"]]
reduced_data.shape


# In[35]:


ks2=range(2,10)


# In[36]:



for n_clusters in ks2:
    
    clusterer = KMeans(n_clusters=n_clusters).fit(reduced_data)

    
    preds = clusterer.predict(reduced_data)

    
    centers = clusterer.cluster_centers_

    
    score = silhouette_score(reduced_data, preds, metric='euclidean')
    print ("For n_clusters = {}. The average silhouette_score is : {}".format(n_clusters, score))


# In[37]:


red_standardized = StandardScaler().fit_transform(reduced_data)


# In[38]:


cluster=KMeans(n_clusters=6,n_init = 12,random_state=0)
cluster.fit(red_standardized)


# In[39]:


y_kmeans1 = cluster.fit_predict(red_standardized)
y_kmeans1


# In[40]:


cluster.cluster_centers_


# In[41]:


frame = pd.DataFrame()
frame['cluster'] = y_kmeans1
frame['cluster'].value_counts()


# In[42]:


df1["Clusters1"]=cluster.fit_predict(red_standardized)


# In[43]:


new_clusterData1=df1[["State","Clusters1","X","Y"]]


# In[44]:


new_clusterData1


# In[45]:


StateClust=new_clusterData1[["State","Clusters1"]]

StateClust=StateClust.sort_values(by=["Clusters1"])
StateClust


# In[46]:


new_clusterData1.set_index('State',inplace=True)


# In[47]:


from sklearn.metrics.pairwise import euclidean_distances


# In[48]:


dists = euclidean_distances(cluster.cluster_centers_)


# In[49]:


import numpy as np
tri_dists = dists[np.triu_indices(6, 1)]
max_dist, avg_dist, min_dist = tri_dists.max(), tri_dists.mean(), tri_dists.min()


# In[50]:


tri_dists


# In[51]:


print(max_dist)
print(avg_dist)
print(min_dist)


# In[52]:


import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from textwrap import dedent


# In[53]:


import io
import xlsxwriter
import flask
from flask import send_file


# In[54]:


import dash_table_experiments as dt
from textwrap import dedent


# In[55]:


label_x = crime_data["State"]
y0 = crime_data["2014"]
y1 = crime_data["2015"]
y2 = crime_data["2016"]
trace1 = go.Scatter(x=label_x, y=y0,mode="lines+markers",fillcolor="seagreen",name="2014")
trace2 = go.Scatter(x=label_x, y=y1,mode="lines+markers",fillcolor="firebrick",name="2015")
trace3 = go.Scatter(x=label_x, y=y2,mode="lines+markers",fillcolor="royalblue",name="2016")


# In[56]:


trace4 = go.Histogram(x=unemployment_data["Unemployment_Urban"], opacity=0.7, name="Urban Unemployment", marker={"line": {"color": "#d62728", "width": 0.2}},xbins={"size": 3}, customdata=unemployment_data["Unemployment_Urban"])
trace5 = go.Histogram(x=unemployment_data["Unemployment_Rural"], opacity=0.7, name="Rural Unemployment", marker={"line": {"color": "#7f7f7f", "width": 0.2}},xbins={"size": 3}, customdata=unemployment_data["Unemployment_Rural"])
data1=[trace4,trace5]


# In[57]:


trace7 = go.Histogram(x=df1["2014"], opacity=0.7, name="2014 Crime Rate", marker={"line": {"color": "#d62728", "width": 0.2}},xbins={"size": 3}, customdata=df1["2014"])
trace8 = go.Histogram(x=df1["2015"], opacity=0.7, name="2015 Crime Rate", marker={"line": {"color": "#7f7f7f", "width": 0.2}},xbins={"size": 3}, customdata=df1["2015"])
trace9 = go.Histogram(x=df1["2016"], opacity=0.7, name="2016 Crime Rate", marker={"line": {"color": "#E1396C", "width": 0.2}},xbins={"size": 3}, customdata=df1["2016"])
data2=[trace7,trace8,trace9]


# In[58]:


trace10 = go.Histogram(x=df1["Suicide Rate (per 1 lakh) 2015[4]"],opacity=0.7, name="Suicide Rate",marker={"line": {"color": "#d62728", "width": 0.2}},xbins={"size": 3},customdata=df1["Suicide Rate (per 1 lakh) 2015[4]"])

                                                          


# In[59]:


fig=go.Figure()


# In[60]:


trace6=go.Scatter(x=suicide_data['State'], y=suicide_data['Suicide Rate (per 1 lakh) 2015[4]'])


# In[61]:


trace11=go.Splom(
                dimensions=[dict(label='Unemployment_Total',
                                 values=df1['Unemployment_Total']),
                            dict(label='Unemployment_Urban',
                                 values=df1['Unemployment_Urban']),
                            dict(label='Unemployment_Rural',
                                 values=df1['Unemployment_Rural']),
                            dict(label='Crime 2014',
                                 values=df1['2014']),
                            dict(label='Crime 2015',
                                 values=df1['2015']),
                            dict(label='Crime 2016',
                                 values=df1['2016']),
                            dict(label='Suicide Rate (per 1 lakh) 2015[4]',
                                 values=df1['Suicide Rate (per 1 lakh) 2015[4]'])],
                text=cols,
                marker=dict(color=['mistyrose', 'moccasin','navajowhite','navy',
                            'oldlace', 'olive', 'olivedrab']
                            ,showscale=False,
                            line_color='white', line_width=0.5)
                )


# In[62]:


var_exp = var
cum_var_exp = np.cumsum(var_exp)

trace12 = dict(
    type='bar',
    x=['PC %s' %i for i in range(1,8)],
    y=var_exp,
    name='Individual'
)

trace13 = dict(
    type='scatter',
    x=['PC %s' %i for i in range(1,8)], 
    y=cum_var_exp,
    name='Cumulative'
)

data3 = [trace12, trace13]


# In[63]:


ks1 =[1,2,3,4,5,6,7,8]
inertias = []
for k in ks1:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)
    
    # Fit model to samples
    model.fit(final_comp.iloc[:,:3])
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
trace14 = go.Scatter(x=ks1, y=inertias,mode="lines+markers",fillcolor="seagreen",name="Variance explained")


# In[64]:


trace15=go.Scatter(x=new_clusterData1[new_clusterData1.Clusters1==0]["X"],
                  y=new_clusterData1[new_clusterData1.Clusters1==0]["Y"],
                  name="clust 1",
                  mode="markers",
                  marker=dict(size=10,color="rgb(188, 189, 34)",
                  line=dict(width=1,color="rgba(0,0,0)")))

trace16=go.Scatter(x=new_clusterData1[new_clusterData1.Clusters1==1]["X"],
                  y=new_clusterData1[new_clusterData1.Clusters1==1]["Y"],
                  name="clust 2",
                  mode="markers",
                  marker=dict(size=10,color="rgb(31, 119, 180)",
                  line=dict(width=1,color="rgba(0,0,0)")))

trace17=go.Scatter(x=new_clusterData1[new_clusterData1.Clusters1==2]["X"],
                  y=new_clusterData1[new_clusterData1.Clusters1==2]["Y"],
                  name="clust 3",
                  mode="markers",
                  marker=dict(size=10,color="rgb(140, 86, 75)",
                  line=dict(width=1,color="rgba(0,0,0)")))

trace18=go.Scatter(x=new_clusterData1[new_clusterData1.Clusters1==3]["X"],
                  y=new_clusterData1[new_clusterData1.Clusters1==3]["Y"],
                  name="clust 4",
                  mode="markers",
                  marker=dict(size=10,color="rgb(145,191,219)",
                  line=dict(width=1,color="rgba(0,0,0)")))

trace19=go.Scatter(x=new_clusterData1[new_clusterData1.Clusters1==4]["X"],
                  y=new_clusterData1[new_clusterData1.Clusters1==4]["Y"],
                  name="clust 5",
                  mode="markers",
                  marker=dict(size=10,color="rgb(255,255,191)",
                  line=dict(width=1,color="rgba(0,0,0)")))
trace20=go.Scatter(x=new_clusterData1[new_clusterData1.Clusters1==5]["X"],
                  y=new_clusterData1[new_clusterData1.Clusters1==5]["Y"],
                  name="clust 6",
                  mode="markers",
                  marker=dict(size=10,color="rgb(252,141,89)",
                  line=dict(width=1,color="rgba(0,0,0)")))

data4=[trace15,trace16,trace17,trace18,trace19,trace20]


# In[65]:


trace21=go.Scatter(x=unemployment_data["State"], y=unemployment_data["Unemployment_Total"])


# In[66]:


import os

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server


app_name = 'My_Dash'


# Boostrap CSS.
#app.css.append_css({'external_url': 'https://cdn.rawgit.com/plotly/dash-app-stylesheets/2d266c578d2a6e8850ebce48fdb52759b2aef506/stylesheet-oil-and-gas.css'})  # noqa: E501

layout = dict(
    autosize=True,
    height=500,
    font=dict(color="#191A1A"),
    titlefont=dict(color="#191A1A", size='14'),
    margin=dict(
        l=35,
        r=35,
        b=35,
        t=45
    )
)

app.layout= html.Div([
    
    html.H1("INDIA -Cities,Crime,Unemployment and Suicide", style={"textAlign": "center"}),
    
    html.Div(  
     dcc.Markdown('''This dash board aims at throwing an overall view on India with regarding its
     **States information(population,literacy and sex ratio with respect to their major cities)**,
     **Crime Rate for the years 2014,2015 & 2016**,
     **Unemployment for the year 2015-16** and
     **Suicide Rate** .
     Finally a Cluster Analysis is also performed for finding out the similarities between the states
     
         '''),style={'color':'black','textAlign':'left'}),

    
    html.H2("Indian States information", style={"textAlign": "center"}),
    html.H2("Data Visualization and Exploratory Data Analysis", style={"textAlign": "left"}),
    
    html.Div([html.Div([dcc.Dropdown(id='Criteria selected1',options=[{'label': i.title(), 'value': i} for i in cities.columns.values[2:]],
    value="population_total")], className="six columns",style={"width": "40%", "float": "right"}),
              
              
              
    html.Div([dcc.Dropdown(id='Criteria selected2',options=[{'label': i.title(), 'value': i} for i in cities.columns.values[2:]],
    value="population_male")], className="six columns", style={"width": "40%", "float": "left"}),
              ], className="row", style={"padding": 50, "width": "60%", "margin-left": "auto", "margin-right": "auto"}),
    dcc.Graph(id='my-graph1'),

    
    
    html.Div([dcc.Graph(id='my-graph2',figure={'data': [trace1,trace2,trace3],'layout':{
                'title':'Crime Rate in Indian States',
                'xaxis':{
                    'title':'States'
                },
                'yaxis':{
                    'title':'Count'
                }
            },},className = "six columns")]),
    

    
    html.Div([
    
       
    html.Div([dcc.Graph(id="my-graph3",figure={'data':data1,'layout':{
                'title':'Urban Vs Rural Unemployment rate',
                'xaxis':{
                    'title':'Unemployment Rate'
                },
                'yaxis':{
                    'title':'Count'
                }
            }})],style={'width': '49%', 'display': 'inline-block', 'vertical-align': 'left'}
    ,className="six columns",),
        
        
    html.Div([dcc.Graph(id="my-graph4",figure={'data':[trace21],
                                             'layout':{
                'title':'Total Unemployment rate in Indian States',
                'xaxis':{
                    'title':'State'
                },
                'yaxis':{
                    'title':'Unemployment Rate'
                }
            }})],style={'width': '49%', 'display': 'inline-block', 'vertical-align': 'left'},className="six columns",)
    
    
    ],className="row",),
    
    
    html.Div([dcc.Graph(id='my-graph5',figure={'data':[trace6],'layout':{
                'title':'Suicide Rate in Indian States',
                'xaxis':{
                    'title':'States'
                },
                'yaxis':{
                    'title':'Count'
                }
            }},className = "six columns")]),
    

    
    html.Div([
    html.Div([dcc.Graph(id="my-graph6",figure={'data':data2,'layout':{
                'title':'Histogram for Crimes in 2014,2015,2016',
                'xaxis':{
                    'title':'Crime Rate'
                },
                'yaxis':{
                    'title':'Count'
                }
            }})],style={'width': '49%', 'display': 'inline-block', 'vertical-align': 'left'}
    ,className="six columns",),
        
    html.Div([dcc.Graph(id="my-graph7",figure={'data':[trace10],'layout':{
                'title':'Histogram for Suicide Rate (per 1 lakh) 2015[4]',
                'xaxis':{
                    'title':'Suicide Rate'
                },
                'yaxis':{
                    'title':'Count'
                }
            }})],style={'width': '49%', 'display': 'inline-block', 'vertical-align': 'right'}
    ,className="six columns",)
        
    ]),
      
    html.Div([
    dcc.Graph(id="my-graph8",figure={'data':[trace11],'layout':{
                'title':'Scatter Plot Matrix',
               }})],className="six columns"
 ),
    html.H2("Analysis for Model Fitting", style={"textAlign": "left"}),
    
    html.Div([
    html.Div([dcc.Graph(id="my-graph9",figure={'data':data3,'layout':{
                'title':'PCA Analysis',
                'xaxis':{
                    'title':'Features'
                },
                'yaxis':{
                    'title':'% Variance'
                }
            }})],style={'width': '49%', 'display': 'inline-block', 'vertical-align': 'left'}
    ,className="six columns",),
    html.Div([dcc.Graph(id="my-graph10",figure={'data':[trace14],'layout':{
                'title':'Elbow Curve',
                'xaxis':{
                    'title':'Number of clusters k'
                },
                'yaxis':{
                    'title':'inertia'
                }
            }})],style={'width': '49%', 'display': 'inline-block', 'vertical-align': 'right'}
    ,className="six columns",),
        
    ]),
    html.H2("KMeans Clustering Output", style={"textAlign": "left"}),
    html.Div([dcc.Graph(id='my-graph11',figure={'data':data4,'layout':{
                'title':'Cluster Analysis',
                
            }},className = "six columns")]),
    html.H2("Final Outcome", style={"textAlign": "left"}),
    
    html.Div([dt.DataTable(rows=StateClust.to_dict('records'),columns=StateClust.columns,row_selectable=True,filterable=False,
    sortable=False,selected_row_indices=[],id='datatable3')],style={'width': '49%', 'display': 'inline-block', 'vertical-align': 'left'}
    ,className="six columns",),
    
        
    
    
    
    
    
    
     ], className="container",style={"border":"2px black solid"})
    
    


@app.callback(
    dash.dependencies.Output('my-graph1', 'figure'),
    [dash.dependencies.Input('Criteria selected1', 'value'),
     dash.dependencies.Input('Criteria selected2', 'value')])
def update_graph(selected_product1, selected_product2):
    dff = cities[(cities[selected_product1] >= 2) & (cities[selected_product2] >= 2)]
    trace1 = go.Bar(x=dff['state_name'], y=dff[selected_product1], name=selected_product1.title(), )
    trace2 = go.Bar(x=dff['state_name'], y=dff[selected_product2], name=selected_product2.title(), )

    return {
        'data': [trace1, trace2],
        'layout': go.Layout(title=f'State Info: {selected_product1.title()}, {selected_product2.title()}',
                            colorway=['#cd7eaf', '#182844'], hovermode="closest",
                            xaxis={'title': "State", 'titlefont': {'color': 'black', 'size': 14},
                                   'tickfont': {'size': 9, 'color': 'black'}},
                            yaxis={'title': "Count", 'titlefont': {'color': 'black', 'size': 14, },
                                   'tickfont': {'color': 'black'}})}


# In[67]:


if __name__ == '__main__':
    app.run_server(debug=False)


# In[ ]:




