import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import scipy.stats as stats
import os

# Define custom color palette for pitch types
pitch_colors = {
    "Fastball": '#ff007d',
    "Four-Seam": '#ff007d',
    "Sinker": "#98165D",
    "Slider": "#67E18D",
    "Sweeper": "#1BB999",
    "Curveball": '#3025CE',
    "ChangeUp": '#F79E70',
    "Splitter": '#90EE32',
    "Cutter": "#BE5FA0",
    "Unknown": '#9C8975',
    "PitchOut": '#472C30'
}

# Define function to load data
@st.cache_data
def load_data():
    # Adjust the path for the deployment environment
    data_file = 'VSGA - Sheet1 (1).csv'
    if not os.path.isfile(data_file):
        st.error(f"Data file {data_file} not found.")
        return pd.DataFrame()  # Return an empty DataFrame if the file is not found

    df = pd.read_csv(data_file)

    # Data transformation similar to R code
    df = df.dropna(subset=["HorzBreak"])
    df['PitchType'] = df['TaggedPitchType'].replace({
        'Four-Seam': 'Fastball', 'Fastball': 'Fastball',
        'Sinker': 'Sinker', 'Slider': 'Slider',
        'Sweeper': 'Sweeper', 'Curveball': 'Curveball',
        'ChangeUp': 'ChangeUp', 'Splitter': 'Splitter',
        'Cutter': 'Cutter'
    }).fillna('Unknown')

    # Format pitcher names and create custom columns
    df['Pitcher'] = df['Pitcher'].str.replace(r'(\w+), (\w+)', r'\2 \1')
    df['inZone'] = np.where((df['PlateLocHeight'].between(1.6, 3.4)) &
                            (df['PlateLocSide'].between(-0.71, 0.71)), 1, 0)
    df['Chase'] = np.where((df['inZone'] == 0) &
                           (df['PitchCall'].isin(['FoulBall', 'FoulBallNotFieldable', 'InPlay', 'StrikeSwinging'])), 1,
                           0)
    df['CustomGameID'] = df['Date'] + ": " + df['AwayTeam'].str[:3] + " @ " + df['HomeTeam'].str[:3]

    return df

# Load data
df = load_data()

if df.empty:
    st.stop()  # Stop execution if no data is loaded

# Sidebar filters
pitcher = st.sidebar.selectbox("Select Pitcher", df['Pitcher'].unique())
games = st.sidebar.multiselect("Select Game(s)", df['CustomGameID'].unique(), default=df['CustomGameID'].unique())
batter_hand = st.sidebar.multiselect("Select Batter Hand", df['BatterSide'].unique(), default=df['BatterSide'].unique())

# Filter data based on user inputs
filtered_data = df[(
                           df['Pitcher'] == pitcher) &
                   (df['CustomGameID'].isin(games)) &
                   (df['BatterSide'].isin(batter_hand))
                   ]

# Calculate metrics
metrics = filtered_data.groupby('PitchType').agg({
    'RelSpeed': 'mean',
    'InducedVertBreak': 'mean',
    'HorzBreak': 'mean',
    'SpinRate': 'mean',
    'RelHeight': 'mean',
    'RelSide': 'mean',
    'Extension': 'mean',
    'VertApprAngle': 'mean'
}).round(2).reset_index()

# Calculate Usage%
total_pitches = len(filtered_data)
usage_percentage = filtered_data['PitchType'].value_counts(normalize=True) * 100
metrics['Usage%'] = metrics['PitchType'].map(usage_percentage).round().astype(int)  # Round and convert to integer

# Reorder columns
metrics = metrics[
    ['PitchType', 'Usage%', 'RelSpeed', 'InducedVertBreak', 'HorzBreak', 'SpinRate', 'RelHeight', 'RelSide',
     'Extension', 'VertApprAngle']]
# Display metrics with Usage% in front of RelSpeed
st.subheader(f"{pitcher}: Pitch Metrics")
st.dataframe(metrics)

# Add strike zone and home plate shape definitions
home_plate = go.Scatter(
    x=[-0.85, 0.85, 0, -0.85],
    y=[0, 0, -0.43, 0],
    mode='lines',
    line=dict(color="Black", width=2),
    name='Home Plate'
)

strike_zone = go.Scatter(
    x=[-0.71, 0.71, 0.71, -0.71, -0.71],
    y=[1.6, 1.6, 3.5, 3.5, 1.6],
    mode='lines',
    line=dict(color="Red", width=2),
    name='Strike Zone'
)
# Plotting Pitch Movement with Plotly
st.subheader(f"{pitcher}: Pitch Movement")
fig = go.Figure()

# Scatter plot for Pitch Movement
fig.add_trace(go.Scatter(
    x=filtered_data['HorzBreak'],
    y=filtered_data['InducedVertBreak'],
    mode='markers',
    marker=dict(color=filtered_data['PitchType'].map(pitch_colors), size=8),
    text=filtered_data['PitchType'],
    hovertemplate='<b>Pitch Type:</b> %{text}<br>' +
                  '<b>Release Speed:</b> %{customdata[0]}<br>' +
                  '<b>Vertical Break:</b> %{customdata[1]}<br>' +
                  '<b>Horizontal Break:</b> %{customdata[2]}<br>' +
                  '<b>Vertical Approach Angle:</b> %{customdata[3]}<br>' +
                  '<b>Pitch Call:</b> %{customdata[4]}<extra></extra>',
    customdata=filtered_data[['RelSpeed', 'InducedVertBreak', 'HorzBreak', 'VertApprAngle', 'PitchCall']].values
))

# Add average breaks as larger, lightly shaded circles
avg_breaks = filtered_data.groupby('PitchType').agg(
    avgHorzBreak=('HorzBreak', 'mean'),
    avgVertBreak=('InducedVertBreak', 'mean')
).reset_index()

for _, row in avg_breaks.iterrows():
    fig.add_trace(go.Scatter(
        x=[row['avgHorzBreak']],
        y=[row['avgVertBreak']],
        mode='markers',
        marker=dict(size=15, color=pitch_colors[row['PitchType']], opacity=0.95, line=dict(color='black', width=2)),
        text=row['PitchType'],
        hovertemplate='<b>Pitch Type:</b> %{text}<br>' +
                      '<b>Horizontal Break:</b> %{x}<br>' +
                      '<b>Vertical Break:</b> %{y}<extra></extra>'
    ))

# Add origin lines
fig.add_shape(
    go.layout.Shape(
        type="line",
        x0=-25,
        y0=0,
        x1=25,
        y1=0,
        line=dict(color="Grey", width=2)
    )
)
fig.add_shape(
    go.layout.Shape(
        type="line",
        x0=0,
        y0=-25,
        x1=0,
        y1=25,
        line=dict(color="Grey", width=2)
    )
)

# Update layout
fig.update_layout(
    title="Pitch Movement (Horizontal vs Vertical Break)",
    xaxis_title="Horizontal Break",
    yaxis_title="Vertical Break",
    xaxis=dict(range=[-25, 25]),
    yaxis=dict(range=[-25, 25])
)

st.plotly_chart(fig)


# Plotting Velocity Distribution using KDE with Plotly
st.subheader(f"{pitcher}: Velocity Distribution (KDE)")
fig = go.Figure()

# Create KDE plot for each PitchType
for pitch_type, color in pitch_colors.items():
    subset = filtered_data[filtered_data['PitchType'] == pitch_type]
    if not subset.empty:
        kde = stats.gaussian_kde(subset['RelSpeed'])
        x = np.linspace(subset['RelSpeed'].min(), subset['RelSpeed'].max(), 1000)
        y = kde(x)
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines',
            name=pitch_type,
            line=dict(color=color)
        ))

# Update layout
fig.update_layout(
    title="Velocity Distribution (Kernel Density Estimate)",
    xaxis_title="Release Speed (mph)",
    yaxis_title="Density"
)

st.plotly_chart(fig)

# Plotting Pitch Locations with Plotly
st.subheader(f"{pitcher}: Pitch Locations")
fig = go.Figure()

# Scatter plot for Pitch Locations
fig.add_trace(go.Scatter(
    x=filtered_data['PlateLocSide'],
    y=filtered_data['PlateLocHeight'],
    mode='markers',
    marker=dict(color=filtered_data['PitchType'].map(pitch_colors), size=8),
    text=filtered_data['PitchType'],
    hovertemplate='<b>Pitch Type:</b> %{text}<br>' +
                  '<b>Release Speed:</b> %{customdata[0]}<br>' +
                  '<b>Vertical Break:</b> %{customdata[1]}<br>' +
                  '<b>Horizontal Break:</b> %{customdata[2]}<br>' +
                  '<b>Vertical Approach Angle:</b> %{customdata[3]}<br>' ,
    customdata=filtered_data[['RelSpeed', 'InducedVertBreak', 'HorzBreak', 'VertApprAngle']].values
))

# Add home plate and strike zone
home_plate = go.Scatter(
    x=[-0.71, 0.71, 0.71, -0.71, -0.71],
    y=[1.6, 1.6, 3.5, 3.5, 1.6],
    mode='lines',
    line=dict(width=2),


)

strike_zone = go.Scatter(
    x=[-0.71, 0.71, 0.71, -0.71, -0.71],
    y=[1.6, 1.6, 3.5, 3.5, 1.6],
    mode='lines',
    line=dict(width=2),

)
fig.add_trace(home_plate)
fig.add_trace(strike_zone)

# Update layout
fig.update_layout(
    title=f"{pitcher}: Pitch Locations",
    xaxis_title="Horizontal Location",
    yaxis_title="Vertical Location",
    xaxis=dict(range=[-2, 2]),
    yaxis=dict(range=[0, 5])
)

st.plotly_chart(fig)

# Plotting Strike Swinging with Plotly
st.subheader(f"{pitcher}: Strike Swinging")
strike_swinging_data = filtered_data[filtered_data['PitchCall'] == 'StrikeSwinging']
fig = go.Figure()

# Scatter plot for Strike Swinging
fig.add_trace(go.Scatter(
    x=filtered_data['PlateLocSide'],
    y=filtered_data['PlateLocHeight'],
    mode='markers',
    marker=dict(color=filtered_data['PitchType'].map(pitch_colors), size=8),
    text=filtered_data['PitchType'],
    hovertemplate='<b>Pitch Type:</b> %{text}<br>' +
                  '<b>Release Speed:</b> %{customdata[0]}<br>' +
                  '<b>Vertical Break:</b> %{customdata[1]}<br>' +
                  '<b>Horizontal Break:</b> %{customdata[2]}<br>' +
                  '<b>Vertical Approach Angle:</b> %{customdata[3]}<br>',
    customdata=filtered_data[['RelSpeed', 'InducedVertBreak', 'HorzBreak', 'VertApprAngle']].values
))


# Add home plate and strike zone
fig.add_trace(home_plate)
fig.add_trace(strike_zone)

# Update layout
fig.update_layout(
    title=f"{pitcher}: Strike Swinging",
    xaxis_title="Horizontal Location",
    yaxis_title="Vertical Location",
    xaxis=dict(range=[-2, 2]),
    yaxis=dict(range=[0, 5])
)

st.plotly_chart(fig)

# Plotting Chase Pitches with Plotly
st.subheader(f"{pitcher}: Chase Pitches")
chase_pitches_data = filtered_data[filtered_data['Chase'] == 1]
fig = go.Figure()

# Scatter plot for Chase Pitches
fig.add_trace(go.Scatter(
    x=filtered_data['PlateLocSide'],
    y=filtered_data['PlateLocHeight'],
    mode='markers',
    marker=dict(color=filtered_data['PitchType'].map(pitch_colors), size=8),
    text=filtered_data['PitchType'],
    hovertemplate='<b>Pitch Type:</b> %{text}<br>' +
                  '<b>Release Speed:</b> %{customdata[0]}<br>' +
                  '<b>Vertical Break:</b> %{customdata[1]}<br>' +
                  '<b>Horizontal Break:</b> %{customdata[2]}<br>' +
                  '<b>Vertical Approach Angle:</b> %{customdata[3]}<br>',
    customdata=filtered_data[['RelSpeed', 'InducedVertBreak', 'HorzBreak', 'VertApprAngle']].values
))

# Add home plate and strike zone
fig.add_trace(home_plate)
fig.add_trace(strike_zone)

# Update layout
fig.update_layout(
    title=f"{pitcher}: Chase Pitches",
    xaxis_title="Horizontal Location",
    yaxis_title="Vertical Location",
    xaxis=dict(range=[-2, 2]),
    yaxis=dict(range=[0, 5])
)

st.plotly_chart(fig)

# Plotting Called Strikes with Plotly
st.subheader(f"{pitcher}: Called Strikes")
called_strikes_data = filtered_data[filtered_data['PitchCall'] == 'StrikeCalled']
fig = go.Figure()

# Scatter plot for Called Strikes
fig.add_trace(go.Scatter(
    x=filtered_data['PlateLocSide'],
    y=filtered_data['PlateLocHeight'],
    mode='markers',
    marker=dict(color=filtered_data['PitchType'].map(pitch_colors), size=8),
    text=filtered_data['PitchType'],
    hovertemplate='<b>Pitch Type:</b> %{text}<br>' +
                  '<b>Release Speed:</b> %{customdata[0]}<br>' +
                  '<b>Vertical Break:</b> %{customdata[1]}<br>' +
                  '<b>Horizontal Break:</b> %{customdata[2]}<br>' +
                  '<b>Vertical Approach Angle:</b> %{customdata[3]}<br>',
    customdata=filtered_data[['RelSpeed', 'InducedVertBreak', 'HorzBreak', 'VertApprAngle']].values
))

# Add home plate and strike zone
fig.add_trace(home_plate)
fig.add_trace(strike_zone)

# Update layout
fig.update_layout(
    title="Called Strikes Locations",
    xaxis_title="Horizontal Location",
    yaxis_title="Vertical Location",
    xaxis=dict(range=[-2, 2]),
    yaxis=dict(range=[0, 5])
)

st.plotly_chart(fig)
