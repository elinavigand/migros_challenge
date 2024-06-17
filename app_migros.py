import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from copy import deepcopy

st.set_page_config(page_title="Migros Challenge", 
                   page_icon="🟧", 
                   #layout="wide", # use full width of the page
                   menu_items={
                       'About': "A Streamlit app to display the best location for next Migros store in Zürich"
                   })

# Data
with open("stzh.adm_stadtkreise_a.json") as response:
    kreise = json.load(response)

with open("stzh.adm_verwaltungsquartiere_bes_p.json") as response:
    quarters = json.load(response)

df_stores =pd.read_csv("data/processed/all_stores_data.csv")
df_all_data =pd.read_csv("data/processed/all_data.csv")

# About the project
st.title("The next Migros store in Zürich")

st.markdown(
    """ 
### Project Objective: 
Identify the optimal location for a new Migros store in Zürich city. 


### Team:
Daniel Karoly, Elina Vigand, Justin Villard, Vahid Mamduhi


### Key factors:

- **Population density:** dense areas offer greater market potential.  
- **Income level:** higher purchasing power indicates a more wealthy customer base.  
- **Market share:** percentage of Migros stores compared to total supermarkets.   
- **Competitor presence:** number of competitor stores in the area.  
    """
    )

st.subheader("Population Density and Median Income Level per Kreis")

# Setting up columns
left_column, middle_column, right_column = st.columns([3, 1, 1])

plot_types = ["Population Density", "Average Income Level"]
plot_type = left_column.radio("Choose Plot Type", plot_types)

# Widgets: selectbox
stores = ["All"]+sorted(pd.unique(df_stores['store']))
s = left_column.selectbox("Choose a store", stores)

# Flow control and plotting
if s == "All":
    reduced_df = df_stores
else:
    reduced_df = df_stores[df_stores["store"] == s]


# Get unique store names and assign a color to each
unique_stores = df_stores['store'].unique()
store_colors = {
    'aldi': '#010160',
    'coop': '#DC3003',
    'lidl': '#FEEF02',
    'migros': '#FF6600'
}

# Create a scatter map for store locations
fig2 = go.Figure()

# Create a choropleth map for district density
fig2.add_trace(
    go.Choroplethmapbox(
        geojson=kreise,
        locations=df_stores['Kreis'],
        featureidkey="properties.name",
        z=df_stores['Kreis Density (population/km^2)'],
        colorscale='Blues',
        colorbar=dict(
            title='District Density',
            y=0.45,   
            len=0.8,
        ),
        hovertemplate="<b>%{location}</b><br>" +
                        "District Density: %{z:.2f}<br>" +
                        "<extra></extra>"
    )
)

# Iterate over unique store names and add traces
for store in unique_stores:
    store_data = reduced_df[reduced_df['store'] == store]

    # Add a scatter trace for each store
    if store == 'migros':
        # Customize markers for 'migros'
        fig2.add_trace(
            go.Scattermapbox(
                ids=store_data['store'],
                lat=store_data['latitude'],
                lon=store_data['longitude'],
                mode='markers',
                opacity=0.7,
                name=store.title(),
                text=store_data.apply(lambda x: f"{x['store'].title()}<br>{x['address']}<br>Kreis {x['Kreis']}", axis=1),
                marker=dict(
                    size=14,
                    color=store_colors[store],
                ),
                hovertemplate="<b>%{text}</b><br><extra></extra>"
            )
        )
    else:
        # Default scatter trace for other stores
        fig2.add_trace(
            go.Scattermapbox(
                ids=store_data['store'],
                lat=store_data['latitude'],
                lon=store_data['longitude'],
                mode='markers',
                name=store.title(),
                text=store_data.apply(lambda x: f"{x['store'].title()}<br>{x['address']}<br>Kreis {x['Kreis']}", axis=1),
                marker=dict(
                    size=10,
                    color=store_colors[store],
                ),
                hovertemplate="<b>%{text}</b><br><extra></extra>"
            )
        )
        
# Create a layout for the map
layout = go.Layout(
    mapbox=dict(
        center={"lat": 47.38, "lon": 8.54},
        style="carto-positron",
        zoom=11.4
    ),
    width=600,
    height=600,
    margin={"r": 0, "t": 35, "l": 0, "b": 0}
)

# Update the layout
fig2.update_layout(layout)

# Show the map
#st.plotly_chart(fig2)


#st.subheader("Median Income Level per Kreis")
# Get unique store names and assign a color to each
unique_stores = df_stores['store'].unique()
store_colors = {
    'aldi': '#010160',
    'coop': '#DC3003',
    'lidl': '#FEEF02',
    'migros': '#FF6600'
}

# Create a scatter map for store locations
fig3 = go.Figure()

# Create a choropleth map for district density
fig3.add_trace(
    go.Choroplethmapbox(
        geojson=kreise,
        locations=df_all_data['Kreis'],
        featureidkey="properties.name",
        z=df_all_data['Median value of taxable income'],
        colorscale='Greens',  # Choose a colorscale
        colorbar=dict(
            title='District Density',
            y=0.45,   
            len=0.8,
        ),
        hovertemplate="<b>%{location}</b><br>" +
                        "Median value of taxable income: %{z:.2f}<br>" +
                        "<extra></extra>"
    )
)


# Iterate over unique store names and add traces
for store in unique_stores:
    store_data = reduced_df[reduced_df['store'] == store]

    # Add a scatter trace for each store
    if store == 'migros':
        # Customize markers for 'migros'
        fig3.add_trace(
            go.Scattermapbox(
                ids=store_data['store'],
                lat=store_data['latitude'],
                lon=store_data['longitude'],
                mode='markers',
                opacity=0.7,
                name=store.title(),
                text=store_data.apply(lambda x: f"{x['store'].title()}<br>{x['address']}<br>Kreis {x['Kreis']}", axis=1),
                marker=dict(
                    size=14,
                    color=store_colors[store],
                ),
                hovertemplate="<b>%{text}</b><br><extra></extra>"
            )
        )
    else:
        # Default scatter trace for other stores
        fig3.add_trace(
            go.Scattermapbox(
                ids=store_data['store'],
                lat=store_data['latitude'],
                lon=store_data['longitude'],
                mode='markers',
                name=store.title(),
                text=store_data.apply(lambda x: f"{x['store'].title()}<br>{x['address']}<br>Kreis {x['Kreis']}", axis=1),
                marker=dict(
                    size=10,
                    color=store_colors[store],
                ),
                hovertemplate="<b>%{text}</b><br><extra></extra>"
            )
        )
        
# Create a layout for the map
layout = go.Layout(
    mapbox=dict(
        center={"lat": 47.38, "lon": 8.54},
        style="carto-positron",
        zoom=11.4
    ),
    width=600,
    height=600,
    margin={"r": 0, "t": 35, "l": 0, "b": 0}
)

# Update the layout
fig3.update_layout(layout)

# Show the map
#st.plotly_chart(fig3)


# Select which plot to show
if plot_type == "Population Density":
    st.plotly_chart(fig2)
else:
    st.plotly_chart(fig3)

# Interactive Location Calculator
st.subheader("Interactive Location Recommender")

# Show procedure

st.markdown(
    """ 
This tool helps you find the best location for a new Migros store in Zürich. Follow these steps:

1. **Understand the key performance indicators (KPIs):**
   - **Migros Market Share (a_1):** Number of Migros stores in a Kreis / Total number of supermarkets (the lower, the better).
   - **Current Population Density (a_2):** Current population density in the Kreis / Population density of Zürich (the higher, the better).
   - **Income Level (a_3):** Average income in the Kreis / Current average income in Zürich (the higher, the better).
   - **Future Population Density (a_4):** Prediction (2040) of population density change / Population density of Zürich (the higher, the better).

2. **Set the weights:**
   - Enter weights for each KPI according to their importance. Default weights are set to 1.00.
   - Adjust weights using the + or - buttons next to each KPI.

3. **Calculate the relevance factor:**
   - The tool uses a weighted linear model: \( \text{Relevance Factor} = a_1 \cdot w_1 + a_2 \cdot w_2 + a_3 \cdot w_3 + a_4 \cdot w_4 \)
   - The weights (w_i) are your input values, and the KPIs (a_i) are predefined.

4. **Get the recommendation:**
   - Based on the weights you set, the tool calculates the relevance factor for each Kreis.
   - The Kreis with the highest factor is the recommended location for the new Migros store.
####
**Example**

- If all weights are set to 1.0, the tool will equally consider all KPIs.
- In the example provided, the recommendation is to open a new Migros store in **Kreis 5**.

**Feel free to adjust the weights and see how the recommendations change based on different priorities.**
####
    """
    )

# First some Data Exploration
@st.cache
def load_data(path):
    df = pd.read_csv(path)
    return df

# Get data for KPIs
Aldi_store_per_kreis=[2, 2, 0, 0, 1, 0, 1, 0, 2, 0, 3, 1]
Lidl_store_per_kreis=[1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 3, 1]
Coop_store_per_kreis=[6, 5, 6, 4, 5, 3, 3, 4, 9, 6, 8, 3]
Migros_store_per_kreis=[4, 4, 3, 3, 3, 2, 3, 2, 3, 2, 3, 1]

Total_store_num=[x_1 + x_2 + x_3 + x_4 for x_1, x_2, x_3, x_4 in zip(
    Aldi_store_per_kreis, Lidl_store_per_kreis, Coop_store_per_kreis, Migros_store_per_kreis)]


# KPI 1: a_1: number of migros stores/total number of stores in a kreis
a_1=[y_1/ y_2 for y_1, y_2 in zip(Migros_store_per_kreis,Total_store_num)]

# KPI 2: a_2: Current population density, normalized by the population density of Zurich
a_2=[0.675171,0.694047,1.212819,2.122174,1.655881,1.458411,0.542626,0.768098,1.005600,0.947106,1.205974,1.144783]

# KPI 3: a_3: Avergae income in a kreis (KCHF)
den_1 = pd.read_csv("./data/processed/average_income_per_kreis_sorted.csv")
mean_income=den_1['Median value of taxable income'].mean()
income_per_kreis=[76.38, 68.53, 58.32, 52.2, 71.07, 74.13, 89.27, 88.97, 57.0, 67.68, 55.3, 44.67]
a_3 = [i /mean_income for i in income_per_kreis]

# KPI 4: a_4: future population density 2040, normalized by the population density of Zurich
future_pop_density=[0.589022, 0.693816,1.206202,1.966494,1.746213,1.396008,0.527892,0.728034,1.000356,0.897879,1.262164,1.307432]
a_4 = [z_1 - z_2 for z_1, z_2 in zip(future_pop_density,a_2)]
a_4_percentage= [i *100 for i in a_4]

# 4 KPIs per district
KPIs = pd.DataFrame(list(zip(np.arange(1,13), a_1, a_2, a_3, a_4)), columns =['Kreis', 'a_1', 'a_2', 'a_3', 'a_4'])

# Weights
# Setting up columns
c1, c2 = st.columns([2, 2])

weights = np.zeros(4)
c1.markdown("**a_1, MIGROS MARKET SHARE:** Number of Migros stores in a Kreis / Total number of supermakets")
weights[0] = -1.0 * c1.number_input("Enter weight w_1:", min_value=0.0, max_value=10.0, value=1.0, step=0.5)

c1.markdown("**a_2, CURRENT POPULATION DENSITY:** Current population density in the Kreis / Population density of Zürich")
weights[1] = c1.number_input("Enter weight w_2:", min_value=0.0, max_value=10.0, value=1.0, step=0.5)

c2.markdown("**a_3, INCOME LEVEL:** Average income in the Kreis / Current average income in Zürich")
weights[2] = c2.number_input("Enter weight w_3:", min_value=0.0, max_value=10.0, value=1.0, step=0.5)

c2.markdown("**a_4, FUTURE POPULATION DENSITY:** Prediction (2040) of population density change / Population density of Zürich")
weights[3] = c2.number_input("Enter weight w_4:", min_value=0.0, max_value=10.0, value=1.0, step=0.5)

# Display the user input
st.markdown(f"Selected weights are w_1 = {weights[0]}, w_2 = {weights[1]}, w_3 = {weights[2]}, w_4 = {weights[3]}.")

# Compute the score per Kreis
relevance_factor = deepcopy(KPIs)
relevance_factor[['w_1a_1', 'w_2a_2', 'w_3a_3', 'w_4a_4']] = relevance_factor[['a_1', 'a_2', 'a_3', 'a_4']].mul(weights)
relevance_factor['score'] = relevance_factor[['w_1a_1', 'w_2a_2', 'w_3a_3', 'w_4a_4']].sum(axis=1)

kreis_recommendation = relevance_factor.rename(columns={'score': 'Relevance Factor', 'a_1':'Migros Market Share (a_1)', 'a_2':'Population Density (a_2)', 
                                                        'a_3':'Normalized AVG Income (a_3)', 'a_4':'Population Change by 2040 (a_4) (%)'})

# Plot the scores on map
with open("./data/raw/stzh.adm_stadtkreise_a.json") as response:
    kreise = json.load(response)

max_factor = kreis_recommendation['Relevance Factor'].max()
df2 = kreis_recommendation.loc[kreis_recommendation['Relevance Factor'] == max_factor]
recom_kreis = df2.values.tolist()[0][0]

st.markdown(f"**Based on the given weights, the recommendation is to consider opening a new Migros store in:**")
message = "Kreis " + str(int(recom_kreis))
markdown_text = f"""
<div style="background-color: lightgray; padding: 10px;">
    <h5>{message}</h5>
</div>
"""
st.markdown(markdown_text, unsafe_allow_html=True)

plotly_map = go.Figure(go.Choroplethmapbox( geojson=kreise, 
                                            locations=relevance_factor.Kreis,
                                            z=relevance_factor.score,
                                            featureidkey="properties.name",
                                            hovertemplate="<b>Kreis: %{location}</b><br>" +
                                            "Relevance Factor: %{z:.2f}<br>" +
                                            "<extra></extra>",
                                            colorscale='Blues',
                                            colorbar=dict(title='Relevance Factor')
                                            ))

plotly_map.update_layout(mapbox_center={"lat": 47.38, "lon": 8.54},
                         mapbox_style="carto-positron", 
                         mapbox_zoom=11.4,
                         width=600,
                         height=600,
                         margin={"r":0,"t":35,"l":0,"b":0},
                         font_color="black",
                         hoverlabel={"bgcolor":"white", 
                                    "font_size":12}
)

st.plotly_chart(plotly_map)

if st.checkbox("Show data"):
    st.dataframe(data=kreis_recommendation)