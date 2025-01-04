# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 20:19:39 2025

@author: Graduate
"""
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import seaborn as sns
import plotly.io as pio
import kaleido

#pio.renderers.default = "browser"

#df = pd.read_csv('2024_pitching_data.csv')
#player_name = 'ryne stanek'
#name_list = player_name.lower().split()
#formatted_name = f"{name_list[1].capitalize()}, {name_list[0].capitalize()}"
#df = df[df['player_name'] == formatted_name]

def plot_pitch_trajectories_with_endpoints_3d(df):
    # Adjust the data for handedness
    df['release_pos_x'] = df.apply(
        lambda row: -row['release_pos_x'] if row['p_throws'] == 'R' else row['release_pos_x'],
        axis=1
    )
    df['ax'] = df.apply(
        lambda row: -row['ax'] if row['p_throws'] == 'L' else row['ax'],
        axis=1
    )
    
    # Group the data by pitch_type and calculate means
    grouped_data = df.groupby('pitch_type').mean()
    palette = sns.color_palette("husl", len(grouped_data))
    
    # Initialize Plotly figure
    fig = go.Figure()

    # Strike zone coordinates
    strike_zone_top = 3.5
    strike_zone_bottom = 1.6
    strike_zone_left = -0.833
    strike_zone_right = 0.833
    strike_zone_front = 1.714
    strike_zone_back = 1.714
    
    # Add ground plane
    fig.add_trace(go.Surface(
        z=np.full((10, 10), strike_zone_bottom-0.5),
        x=np.linspace(-5, 5, 10),
        y=np.linspace(8, 45, 10),
        colorscale=[[0, 'green'], [1, 'green']],
        opacity=1.0,
        showscale=False,
        showlegend=False
    ))
    
    # Add mound plane
    fig.add_trace(go.Surface(
        z=np.full((10, 10), strike_zone_bottom-0.5),
        x=np.linspace(-5, 5, 10),
        y=np.linspace(45, 75, 10),
        colorscale=[[0, 'brown'], [1, 'brown']],
        opacity=1.0,
        showscale=False,
        showlegend=False
    ))
    
    # Add batter's box plane
    fig.add_trace(go.Surface(
        z=np.full((10, 10), strike_zone_bottom-0.5),
        x=np.linspace(-5, 5, 10),
        y=np.linspace(0, 8, 10),
        colorscale=[[0, 'brown'], [1, 'brown']],
        opacity=1.0,
        showscale=False,
        showlegend=False
    ))

    # Add strike zone as a wireframe
    for z in [strike_zone_top, strike_zone_bottom]:
        fig.add_trace(go.Scatter3d(
            x=[strike_zone_left, strike_zone_right, strike_zone_right, strike_zone_left, strike_zone_left],
            y=[strike_zone_front, strike_zone_front, strike_zone_back, strike_zone_back, strike_zone_front],
            z=[z, z, z, z, z],
            mode='lines',
            line=dict(color='black', width=4),
            showlegend=False
        ))
    for x, y in zip(
        [strike_zone_left, strike_zone_right, strike_zone_left, strike_zone_right],
        [strike_zone_front, strike_zone_front, strike_zone_back, strike_zone_back]
    ):
        fig.add_trace(go.Scatter3d(
            x=[x, x],
            y=[y, y],
            z=[strike_zone_bottom, strike_zone_top],
            mode='lines',
            line=dict(color='black', width=4),
            showlegend=False
        ))

    legend_labels = []

    # Plot each pitch type
    for i, (pitch_type, row) in enumerate(grouped_data.iterrows()):
        release_pos_x = row['release_pos_x']
        release_pos_y = row['release_pos_y']
        release_pos_z = row['release_pos_z']
        v_x = row['vx0']
        v_y = row['vy0']
        v_z = row['vz0']
        a_x = row['ax']
        a_y = row['ay']
        a_z = row['az']

        # Time steps
        t_final = 0.6
        n_steps = 1000
        t = np.linspace(0, t_final, n_steps)

        # Calculate positions
        x = release_pos_x + v_x * t + 0.5 * a_x * t**2
        y = release_pos_y + v_y * t + 0.5 * a_y * t**2
        z = release_pos_z + v_z * t + 0.5 * a_z * t**2

        # Filter points where y > strike zone front
        mask = y >= strike_zone_front
        x = x[mask]
        y = y[mask]
        z = z[mask]

        color = f"rgb({palette[i][0]*255}, {palette[i][1]*255}, {palette[i][2]*255})"

        # Add trajectory line
        fig.add_trace(go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='lines',
            line=dict(color=color, width=8),
            name=pitch_type
        ))

        # Add start and end points
        fig.add_trace(go.Scatter3d(
            x=[release_pos_x, x[-1]],
            y=[release_pos_y, y[-1]],
            z=[release_pos_z, z[-1]],
            mode='markers',
            marker=dict(size=6, color=color),
            showlegend=False
        ))

        # Keep track of legend labels
        if pitch_type not in legend_labels:
            legend_labels.append(pitch_type)

    fig.update_layout(
    title=dict(
        text="3D Pitch Trajectory",
        x=0.5,  # Center the title horizontally
        xanchor='center',  # Ensure centering alignment
        yanchor='top',  # Optional: adjust vertical alignment if needed
        font=dict(size=20)  # Optional: adjust font size
    ),
    scene=dict(
        xaxis=dict(range=[-4, 4], showticklabels=False, title=''),
        yaxis=dict(range=[1, 60], showticklabels=False, title=''),
        zaxis=dict(range=[0.8, 7], showticklabels=False, title=''),
        aspectratio=dict(x=1, y=12, z=1),
        camera=dict(
            eye=dict(x=0, y=-6.45, z=-0.25)
        ),
    ),
    legend=dict(
        x=0,  # Move the legend to the left (horizontal position)
        y=1,  # Vertical position (top of the plot)
        xanchor="left",  # Anchor the legend to the left
        yanchor="top",  # Anchor the legend to the top
    ),
    legend_title=dict(text="Pitch Type"),
)
    
    fig.write_image("3d_plot.png", width=800, height=600)
    import cv2

    def crop_func(image_path, output_path):
        # Read the image
        img = cv2.imread(image_path)

        # Get the image dimensions
        height, width, _ = img.shape

        # Crop 75 pixels from each side (left, right, top, bottom)
        cropped_img = img[75:height-75, 75:width-75]

        # Save the cropped image
        cv2.imwrite(output_path, cropped_img)
    
    crop_func("3d_plot.png", "cropped_3d_plot.png")

#plot_pitch_trajectories_with_endpoints_3d(filtered_data.copy())
