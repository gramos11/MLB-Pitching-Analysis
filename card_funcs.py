# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 13:48:06 2025

@author: Graduate
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Ellipse
from matplotlib.font_manager import FontProperties
from matplotlib.patheffects import withStroke
import matplotlib.gridspec as gridspec
from pybaseball import playerid_lookup
from pybaseball import statcast_pitcher
import matplotlib as mpl
from matplotlib.lines import Line2D
import matplotlib.image as mpimg
import cv2
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import requests
from io import BytesIO
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import plotly_test

### A portion of the following code, mostly for formatting, and scraping, was taken from
### Thomas Nestico at https://github.com/tnestico/pitching_summary

mlb_teams = [
    {"team": "AZ", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/ari.png&h=500&w=500"},
    {"team": "ATL", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/atl.png&h=500&w=500"},
    {"team": "BAL", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/bal.png&h=500&w=500"},
    {"team": "BOS", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/bos.png&h=500&w=500"},
    {"team": "CHC", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/chc.png&h=500&w=500"},
    {"team": "CWS", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/chw.png&h=500&w=500"},
    {"team": "CIN", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/cin.png&h=500&w=500"},
    {"team": "CLE", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/cle.png&h=500&w=500"},
    {"team": "COL", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/col.png&h=500&w=500"},
    {"team": "DET", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/det.png&h=500&w=500"},
    {"team": "HOU", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/hou.png&h=500&w=500"},
    {"team": "KC", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/kc.png&h=500&w=500"},
    {"team": "LAA", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/laa.png&h=500&w=500"},
    {"team": "LAD", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/lad.png&h=500&w=500"},
    {"team": "MIA", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/mia.png&h=500&w=500"},
    {"team": "MIL", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/mil.png&h=500&w=500"},
    {"team": "MIN", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/min.png&h=500&w=500"},
    {"team": "NYM", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/nym.png&h=500&w=500"},
    {"team": "NYY", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/nyy.png&h=500&w=500"},
    {"team": "OAK", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/oak.png&h=500&w=500"},
    {"team": "PHI", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/phi.png&h=500&w=500"},
    {"team": "PIT", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/pit.png&h=500&w=500"},
    {"team": "SD", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/sd.png&h=500&w=500"},
    {"team": "SF", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/sf.png&h=500&w=500"},
    {"team": "SEA", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/sea.png&h=500&w=500"},
    {"team": "STL", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/stl.png&h=500&w=500"},
    {"team": "TB", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/tb.png&h=500&w=500"},
    {"team": "TEX", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/tex.png&h=500&w=500"},
    {"team": "TOR", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/tor.png&h=500&w=500"},
    {"team": "WSH", "logo_url": "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/wsh.png&h=500&w=500"}
]
df_image = pd.DataFrame(mlb_teams)
image_dict = df_image.set_index('team')['logo_url'].to_dict()

stats_dict = {'IP':{'table_header':'$\\bf{IP}$','format':'.1f',} ,
 'TBF':{'table_header':'$\\bf{PA}$','format':'.0f',} ,
 'AVG':{'table_header':'$\\bf{AVG}$','format':'.3f',} ,
 'K/9':{'table_header':'$\\bf{K\/9}$','format':'.2f',} ,
 'BB/9':{'table_header':'$\\bf{BB\/9}$','format':'.2f',} ,
 'K/BB':{'table_header':'$\\bf{K\/BB}$','format':'.2f',} ,
 'HR/9':{'table_header':'$\\bf{HR\/9}$','format':'.2f',} ,
 'K%':{'table_header':'$\\bf{K\%}$','format':'.1%',} ,
 'BB%':{'table_header':'$\\bf{BB\%}$','format':'.1%',} ,
 'K-BB%':{'table_header':'$\\bf{K-BB\%}$','format':'.1%',} ,
 'WHIP':{'table_header':'$\\bf{WHIP}$','format':'.2f',} ,
 'BABIP':{'table_header':'$\\bf{BABIP}$','format':'.3f',} ,
 'LOB%':{'table_header':'$\\bf{LOB\%}$','format':'.1%',} ,
 'xFIP':{'table_header':'$\\bf{xFIP}$','format':'.2f',} ,
 'FIP':{'table_header':'$\\bf{FIP}$','format':'.2f',} ,
 'H':{'table_header':'$\\bf{H}$','format':'.0f',} ,
 '2B':{'table_header':'$\\bf{2B}$','format':'.0f',} ,
 '3B':{'table_header':'$\\bf{3B}$','format':'.0f',} ,
 'R':{'table_header':'$\\bf{R}$','format':'.0f',} ,
 'ER':{'table_header':'$\\bf{ER}$','format':'.0f',} ,
 'HR':{'table_header':'$\\bf{HR}$','format':'.0f',} ,
 'BB':{'table_header':'$\\bf{BB}$','format':'.0f',} ,
 'IBB':{'table_header':'$\\bf{IBB}$','format':'.0f',} ,
 'HBP':{'table_header':'$\\bf{HBP}$','format':'.0f',} ,
 'SO':{'table_header':'$\\bf{SO}$','format':'.0f',} ,
 'OBP':{'table_header':'$\\bf{OBP}$','format':'.0f',} ,
 'SLG':{'table_header':'$\\bf{SLG}$','format':'.0f',} ,
 'ERA':{'table_header':'$\\bf{ERA}$','format':'.2f',} ,
 'wOBA':{'table_header':'$\\bf{wOBA}$','format':'.3f',} ,
 'G':{'table_header':'$\\bf{G}$','format':'.0f',} }

def get_data(player_name='emmanuel clase'):
    name_list = player_name.lower().split()
    first_name = name_list[0]
    last_name = name_list[1]
    table = playerid_lookup(last_name, first_name)
    key = table['key_mlbam'][0]
    table2 = statcast_pitcher('2024-03-28', '2024-11-01', key)
    table2 = table2.dropna(subset=["pfx_x", "pfx_z", "p_throws"])
    if table2["p_throws"].iloc[0] == "L":
        table2["pfx_x"] = -table2["pfx_x"]
    
    table2["pfx_x"] = table2["pfx_x"] * 12
    table2["pfx_z"] = table2["pfx_z"] * 12
    table3 = table2.groupby(["pitch_type", "game_date"])[["pfx_x", "pfx_z"]].mean().reset_index()

    stuff_data = pd.read_csv("best_pitches_2024.csv")
    stuff_data['player_name'] = stuff_data['player_name'].apply(lambda x: ' '.join(x.split(', ')[::-1]))  # Convert "Last, First" to "First Last"
    player_data = stuff_data[stuff_data['player_name'] == f"{first_name.capitalize()} {last_name.capitalize()}"]

    data = pd.read_csv('py_pitch_data_2020_2024.csv')
    filtered_data = data[data['game_year'] == 2024]
    formatted_name = f"{name_list[1].capitalize()}, {name_list[0].capitalize()}"
    filtered_data = filtered_data[filtered_data['player_name'] == formatted_name]
    
    stats = ['IP','TBF','WHIP','ERA', 'FIP', 'K%', 'BB%', 'K-BB%']
    
    return key, table2, table3, stuff_data, player_data, filtered_data, stats

def plot_confidence_ellipse(x, y, ax, color, n_std=1.0, ellipse_alpha=0.3, **kwargs):
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    mean_x, mean_y = np.mean(x), np.mean(y)
    cov = np.cov(x, y)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    width, height = 2 * n_std * np.sqrt(eigenvalues)
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))

    ellipse = Ellipse(
        (mean_x, mean_y), width, height, angle, edgecolor=color, facecolor=color, alpha=ellipse_alpha, **kwargs
    )
    ax.add_patch(ellipse)
    
def plot_pitch_break(df, df2, ax):
    # Plot scatter points and ellipses for each pitch_type
    palette = sns.color_palette("husl", len(df["pitch_type"].unique()))
    color_mapping = dict(zip(df["pitch_type"].unique(), palette))

    for pitch_type, group in df.groupby("pitch_type"):
        ax.scatter(group["pfx_x"], group["pfx_z"], label=pitch_type,
                   color=color_mapping[pitch_type], alpha=0.7, edgecolor='black', s=100)
    for pitch_type, group in df2.groupby("pitch_type"):
        plot_confidence_ellipse(group["pfx_x"], group["pfx_z"], ax,
                                color=color_mapping[pitch_type], ellipse_alpha=0.3)

    ax.set_xlim(-22, 22)
    ax.set_ylim(-22, 22)
    ax.axhline(0, color='orange', linestyle='--', linewidth=1)  # Horizontal line (y=0)
    ax.axvline(0, color='orange', linestyle='--', linewidth=1)
    ax.set_xlabel("Horizontal Break (inches)")
    ax.set_ylabel("Induced Vertical Break (inches)")
    ax.set_title("Pitch Break")
    ax.grid(True)
    ax.legend(title="Pitch Type")
    
    
def plot_stuff_plus(p_data, s_data, ax):
    # Ensure 'pitch_type' is a valid column and contains no missing values
    if 'pitch_type' not in s_data.columns or 'pitch_type' not in p_data.columns:
        raise ValueError("'pitch_type' column is missing from one of the DataFrames")

    if s_data['pitch_type'].isnull().any() or p_data['pitch_type'].isnull().any():
        raise ValueError("There are missing values in 'pitch_type' columns")

    # Get unique pitch types from the data that have actual data points
    pitch_types = s_data['pitch_type'].unique()
    valid_pitch_types = [pt for pt in pitch_types if not p_data[p_data['pitch_type'] == pt].empty]

    # If no valid pitch types, print a message and return
    if not valid_pitch_types:
        print("No valid pitch types to plot.")
        return

    # Create color palette for each valid pitch type
    palette = sns.color_palette("husl", n_colors=len(valid_pitch_types))
    
    s_data['pitch_type'] = pd.Categorical(s_data['pitch_type'], categories=valid_pitch_types, ordered=True)
    p_data['pitch_type'] = pd.Categorical(p_data['pitch_type'], categories=valid_pitch_types, ordered=True)

    # Plot data for each valid pitch type
    for pitch_type, color in zip(valid_pitch_types, palette):
        # Filter data for each pitch type
        pitch_data = s_data[s_data['pitch_type'] == pitch_type]
        player_pitch = p_data[p_data['pitch_type'] == pitch_type]

        # Plot swarm plot for each pitch type on the same axis
        sns.swarmplot(
            data=pitch_data, 
            x="pitch_type",  # Keep x as the categorical 'pitch_type'
            y="avg_tj_stuff_plus", 
            color=color, 
            size=5, 
            alpha=0.3,  # Increased translucency of data points
            ax=ax
        )

        # Highlight the player's pitch grade with larger and more opaque dots
        for _, row in player_pitch.iterrows():
            ax.scatter(
                row['pitch_type'], row['avg_tj_stuff_plus'],  # Keep x as categorical 'pitch_type'
                color=color, s=800, edgecolor='black', alpha=0.8,  # More opaque for the player's dot
                zorder=5
            )
            # Dynamically adjust font size and place text inside the dot
            pitch_grade = round(row['avg_tj_stuff_plus'])
            font_properties = FontProperties(
                family='Arial',
                weight='heavy',
                size=12
            )
            ax.text(
                row['pitch_type'], row['avg_tj_stuff_plus'], 
                f"{pitch_grade}",  # Round pitch grade to nearest whole number
                color='white', ha='center', va='center', fontproperties=font_properties, path_effects=[withStroke(linewidth=2, foreground='black')], zorder=10
            )

    # Add horizontal line at 100 (midpoint)
    ax.axhline(100, color='gray', linestyle='--', linewidth=1)

    # Set subplot title and limits
    ax.set_title("Stuff+ by Pitch Type", fontsize=16)
    ax.set_ylim(70, 120)
    
    # Set x-ticks to represent each valid pitch type
    ax.set_xticks(range(len(valid_pitch_types)))
    ax.set_xticklabels(valid_pitch_types)  # Label each valid pitch type

    ax.set_xlabel("Pitch Type", fontsize=12)
    ax.set_ylabel("Stuff+", fontsize=12)
    
def plot_pitch_location(df, ax):
    palette = sns.color_palette("husl", len(df["pitch_type"].unique()))
    color_mapping = dict(zip(df["pitch_type"].unique(), palette))
    
    values = list(color_mapping.values())
    sorted_keys = sorted(color_mapping.keys())
    sorted_color_mapping = dict(zip(sorted_keys, values))

    pitch_counts = df['pitch_type'].value_counts()
    pitch_counts = pitch_counts.sort_index()
    
    for pitch_type, group in df.groupby('pitch_type'):
        x = group['plate_x']
        y = group['plate_z']
        color = sorted_color_mapping[pitch_type]
        frequency = pitch_counts[pitch_type]  # Frequency of the pitch type

        # Scale the size of the mean marker based on frequency
        marker_size = 20 + (frequency * 5)  # Adjust size scaling as needed

        # Plot confidence ellipse
        plot_confidence_ellipse(x, y, ax, color=color, n_std=1.0, ellipse_alpha=0.3)
        # Plot mean point with size based on frequency
        ax.scatter(np.mean(x), np.mean(y), color=color, edgecolor='black', s=marker_size, zorder=5)
    
    # Strike zone dimensions
    strike_zone_top = 3.5  # Top of the strike zone
    strike_zone_bottom = 1.5  # Bottom of the strike zone
    strike_zone_left = -0.83  # Left side of the strike zone
    strike_zone_right = 0.83  # Right side of the strike zone

    # Draw the strike zone rectangle
    strike_zone = plt.Rectangle(
        (strike_zone_left, strike_zone_bottom),  # Bottom-left corner
        strike_zone_right - strike_zone_left,  # Width
        strike_zone_top - strike_zone_bottom,  # Height
        edgecolor='black',
        facecolor='none',
        linewidth=2
    )
    ax.add_patch(strike_zone)

    # Set plot limits and labels
    ax.set_xlim(-2, 2)
    ax.set_ylim(0, 4)

    # Remove x and y axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Turn off spines if desired
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Add grid and legend
    ax.grid(False)
    ax.legend(handles=[plt.Line2D([0], [0], marker='o', color=color, label=f"{pitch_type} ({pitch_counts[pitch_type]})", 
                                  linestyle='', markersize=10) 
                       for pitch_type, color in sorted_color_mapping.items()], 
              title="Pitch Type (Frequency)", loc="upper right")
    
    
def plot_pitch_trajectories_with_endpoints_3d(df, ax=None):
    df['release_pos_x'] = df.apply(
        lambda row: -row['release_pos_x'] if row['p_throws'] == 'R' else row['release_pos_x'],
        axis=1
    )
    
    df['ax'] = df.apply(
        lambda row: -row['ax'] if row['p_throws'] == 'L' else row['ax'],
        axis=1
    )
    # Group the data by pitch_type and calculate means for required variables
    grouped_data = df.groupby('pitch_type').mean()
    
    # Initialize the 3D figure and axes
    if ax is None:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
                
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1.0, 12.0, 1.0, 1]))
    
    # Define the coordinates for the top and bottom corners of the strike zone
    strike_zone_top = 3.5
    strike_zone_bottom = 1.6
    strike_zone_left = -0.833
    strike_zone_right = 0.833
    strike_zone_front = 1.714  # Fixed y-coordinate for both top and bottom corners
    
    # Add ground plane (green grass)
    x_ground, y_ground = np.meshgrid(np.linspace(-5, 5, 10), np.linspace(40, 75, 10))
    z_ground = np.zeros_like(x_ground)
    ax.plot_surface(x_ground, y_ground, z_ground, color='green', alpha=0.5)
    
    # Add brown dirt for the pitcher's mound
    x_dirt, y_dirt = np.meshgrid(np.linspace(-5, 5, 10), np.linspace(85, 110, 10))
    z_dirt = np.zeros_like(x_dirt) + 0.02  # Slight elevation
    ax.plot_surface(x_dirt, y_dirt - 12, z_dirt, color='brown', alpha=0.5)
    
    # Add brown dirt for the batter's box
    x_box, y_box = np.meshgrid(np.linspace(-5, 5, 10), np.linspace(20, 52, 10))
    z_box = np.zeros_like(x_box)
    ax.plot_surface(x_box, y_box - 12, z_box, color='brown', alpha=0.5)
    
    # Coordinates for the four corners of the strike zone
    top_left = (strike_zone_left, strike_zone_front, strike_zone_top)
    top_right = (strike_zone_right, strike_zone_front, strike_zone_top)
    bottom_left = (strike_zone_left, strike_zone_front, strike_zone_bottom)
    bottom_right = (strike_zone_right, strike_zone_front, strike_zone_bottom)
    
    # Create the wireframe for the strike zone
    strike_zone_x = [top_left[0], top_right[0], top_right[0], top_left[0], top_left[0], bottom_left[0], bottom_right[0], bottom_right[0], bottom_left[0], top_left[0]]
    strike_zone_y = [top_left[1], top_right[1], top_right[1], top_left[1], top_left[1], bottom_left[1], bottom_right[1], bottom_right[1], bottom_left[1], top_left[1]]
    strike_zone_z = [top_left[2], top_right[2], bottom_right[2], bottom_left[2], top_left[2], bottom_left[2], bottom_right[2], bottom_right[2], bottom_left[2], top_left[2]]
    
    palette = sns.color_palette("husl", len(grouped_data))
    legend_handles = []

    # Iterate through each pitch type and plot its trajectory
    for i, (pitch_type, row) in enumerate(grouped_data.iterrows()):
        # Extract average variables for this pitch type
        release_pos_x = row['release_pos_x']
        release_pos_y = row['release_pos_y']
        release_pos_z = row['release_pos_z']
        v_x = row['vx0']
        v_y = row['vy0']  # Added y-dimension
        v_z = row['vz0']
        a_x = row['ax']
        a_y = row['ay']  # Added y-dimension
        a_z = row['az']
        
        value = ((v_y**2) - 2 * a_y * (50 - release_pos_y))**0.5
        tr = (-v_y - value)/a_y
        
        v_y0 = v_y + a_y * tr
        v_x0 = v_x + a_x * tr
        v_z0 = v_z + a_z * tr
        
        #plate_x = row['plate_x']
        #plate_z = row['plate_z']
        #plate_y = 0

        # Time steps
        t_final = 0.6
        n_steps = 1000
        t = np.linspace(0, t_final, n_steps)

        # Calculate positions
        x = release_pos_x + v_x0 * t + 0.5 * a_x * t**2
        y = release_pos_y + v_y0 * t + 0.5 * a_y * t**2
        z = release_pos_z + v_z0 * t + 0.5 * a_z * t**2
        
        # Filter points where y > 0 (before the pitch reaches the plate)
        mask = y >= 1.714  # Only keep points where y >= 1.714
        x = x[mask]
        y = y[mask]
        z = z[mask]
        
        color = palette[i]

        # Plot the trajectory
        ax.plot(x, y, z, color=color, linewidth=2, alpha=1)

        # Mark the start and end points
        ax.scatter(release_pos_x, release_pos_y, release_pos_z, marker='o', color=color, s=100, zorder=5, alpha=0.6)  # Start point
        #ax.scatter(x[-1], y[-1], z[-1], marker='x', color=color, s=100, zorder=5)  # End point
        ax.scatter(x[-1], y[-1], z[-1], marker='x', color=color, s=100, zorder=5)  # End point at the plate
        # Plot the wireframe for the strike zone
        ax.plot(strike_zone_x, strike_zone_y, strike_zone_z, color='black', linewidth=2)
        # Add the pitch_type to the legend
        if i == 0 or pitch_type != grouped_data.index[i-1]:
            legend_handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=pitch_type))
    
    ax.view_init(elev=0, azim=-90)
    # Format the plot
    ax.set_xlim(-4, 4)
    ax.set_ylim(1, 60)  # Adjust y-limits as needed
    ax.set_zlim(0.8, 7)
    #ax.set_title(f"3D Pitch Trajectories for {formatted_name}")
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    
    ax.legend(handles=legend_handles, title="Pitch Type", loc="upper center")
    plt.savefig('trajectory_plot.png')
    # Show the plot
    plt.show()

    def crop_whitespace_from_plot(image_path, output_path):
        # Read the image
        img = cv2.imread(image_path)

        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply a binary threshold to get a binary image where white becomes 255 and everything else is 0
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

        # Invert the binary image (so white areas become black)
        thresh = cv2.bitwise_not(thresh)

        # Find contours (external contours to locate the content area)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get the bounding box of the largest contour (the plot content)
        x, y, w, h = cv2.boundingRect(contours[0])

        # Crop the image using the bounding box
        cropped_img = img[y:y+h, x:x+w]

        # Save or display the cropped image
        cv2.imwrite(output_path, cropped_img)
        
    crop_whitespace_from_plot("trajectory_plot.png", "cropped_trajectory_plot.png")
    
## Thomas Nestico
def player_headshot(pitcher_id, ax):
    # Construct the URL for the player's headshot image
    url = f'https://img.mlbstatic.com/mlb-photos/image/'\
          f'upload/d_people:generic:headshot:67:current.png'\
          f'/w_640,q_auto:best/v1/people/{pitcher_id}/headshot/silo/current.png'

    # Send a GET request to the URL
    response = requests.get(url)

    # Open the image from the response content
    img = Image.open(BytesIO(response.content))


    # Display the image on the axis
    ax.set_xlim(0, 1.3)
    ax.set_ylim(0, 1)
    ax.imshow(img, extent=[0, 1, 0, 1], origin='upper')

    # Turn off the axis
    ax.axis('off')

## Thomas Nestico
def player_bio(pitcher_id, ax):
    # Construct the URL to fetch player data
    url = f"https://statsapi.mlb.com/api/v1/people?personIds={pitcher_id}&hydrate=currentTeam"

    # Send a GET request to the URL and parse the JSON response
    data = requests.get(url).json()

    # Extract player information from the JSON data
    player_name = data['people'][0]['fullName']
    pitcher_hand = data['people'][0]['pitchHand']['code']
    age = data['people'][0]['currentAge']
    height = data['people'][0]['height']
    weight = data['people'][0]['weight']

    # Display the player's name, handedness, age, height, and weight on the axis
    ax.text(0.5, 1, f'{player_name}', va='top', ha='center', fontsize=56)
    ax.text(0.5, 0.65, f'{pitcher_hand}HP, Age:{age}, {height}/{weight}', va='top', ha='center', fontsize=30)
    ax.text(0.5, 0.40, f'Season Pitching Summary', va='top', ha='center', fontsize=40)
    ax.text(0.5, 0.15, f'2024 MLB Season', va='top', ha='center', fontsize=30, fontstyle='italic')

    # Turn off the axis
    ax.axis('off')
    
## Thomas Nestico
def plot_logo(pitcher_id: str, image_dict, ax: plt.Axes):
    # Construct the URL to fetch player data
    url = f"https://statsapi.mlb.com/api/v1/people?personIds={pitcher_id}&hydrate=currentTeam"

    # Send a GET request to the URL and parse the JSON response
    data = requests.get(url).json()

    # Construct the URL to fetch team data
    url_team = 'https://statsapi.mlb.com/' + data['people'][0]['currentTeam']['link']

    # Send a GET request to the team URL and parse the JSON response
    data_team = requests.get(url_team).json()

    # Extract the team abbreviation
    team_abb = data_team['teams'][0]['abbreviation']

    # Get the logo URL from the image dictionary using the team abbreviation
    logo_url = image_dict[team_abb]

    # Send a GET request to the logo URL
    response = requests.get(logo_url)

    # Open the image from the response content
    img = Image.open(BytesIO(response.content))

    # Display the image on the axis
    ax.set_xlim(0, 1.3)
    ax.set_ylim(0, 1)
    ax.imshow(img, extent=[0.3, 1.3, 0, 1], origin='upper')

    # Turn off the axis
    ax.axis('off')

## Thomas Nestico
def fangraphs_pitching_leaderboards(season):
    url = f"https://www.fangraphs.com/api/leaders/major-league/data?age=&pos=all&stats=pit&lg=all&season={season}&season1={season}&ind=0&qual=0&type=8&month=0&pageitems=500000"
    data = requests.get(url).json()
    df = pd.DataFrame(data=data['data'])
    return df
## Thomas Nestico
def fangraphs_pitcher_stats(pitcher_id, ax, stats_dict, stats, season, fontsize=20):
    df_fangraphs = fangraphs_pitching_leaderboards(season = season)

    df_fangraphs_pitcher = df_fangraphs[df_fangraphs['xMLBAMID'] == pitcher_id][stats].reset_index(drop=True)

    df_fangraphs_pitcher.loc[0] = [format(df_fangraphs_pitcher[x][0],stats_dict[x]['format']) if df_fangraphs_pitcher[x][0] != '---' else '---' for x in df_fangraphs_pitcher]
    table_fg = ax.table(cellText=df_fangraphs_pitcher.values, colLabels=stats, cellLoc='center',
                    bbox=[0.00, 0.0, 1, 1])

    table_fg.set_fontsize(fontsize)


    new_column_names = [stats_dict[x]['table_header'] if x in df_fangraphs_pitcher else '---' for x in stats]
    # #new_column_names = ['Pitch Name', 'Pitch%', 'Velocity', 'Spin Rate','Exit Velocity', 'Whiff%', 'CSW%']
    for i, col_name in enumerate(new_column_names):
        table_fg.get_celld()[(0, i)].get_text().set_text(col_name)

    ax.axis('off')
    
## Thomas Nestico (Formatting)
def pitching_dashboard(pitcher_id, table3, table2, filtered_data, p_data, s_data, stats):
    # Create a 20 by 20 figure
    #df = df_processing(df)
    fig = plt.figure(figsize=(20, 20))

    # Create a gridspec layout with 8 columns and 6 rows
    # Include border plots for the header, footer, left, and right
    gs = gridspec.GridSpec(6, 8,
                        height_ratios=[2,20,9,36,36,7],
                        width_ratios=[1,18,18,18,18,18,18,1])

    # Define the positions of each subplot in the grid
    ax_headshot = fig.add_subplot(gs[1,1:3])
    ax_bio = fig.add_subplot(gs[1,3:5])
    ax_logo = fig.add_subplot(gs[1,5:7])

    ax_season_table = fig.add_subplot(gs[2,1:7])

    ax_plot_1 = fig.add_subplot(gs[3,1:3])
    ax_plot_1.set_aspect('equal')

    ax_plot_2 = fig.add_subplot(gs[3,3:5])
    ax_plot_3 = fig.add_subplot(gs[3,5:7])

    ax_table = fig.add_subplot(gs[4,1:7])

    ax_footer = fig.add_subplot(gs[-1,1:7])
    ax_header = fig.add_subplot(gs[0,1:7])
    ax_left = fig.add_subplot(gs[:,0])
    ax_right = fig.add_subplot(gs[:,-1])

    # Hide axes for footer, header, left, and right
    ax_footer.axis('off')
    ax_header.axis('off')
    ax_left.axis('off')
    ax_right.axis('off')

    # Call the functions
    fangraphs_pitcher_stats(pitcher_id, ax_season_table, stats_dict, stats, season=2024, fontsize=20)

    player_headshot(pitcher_id, ax=ax_headshot)
    player_bio(pitcher_id, ax=ax_bio)
    plot_logo(pitcher_id, image_dict, ax=ax_logo)
    
    plot_pitch_break(table3, table2, ax_plot_3)
    plot_stuff_plus(p_data, s_data, ax_table)
    plot_pitch_location(filtered_data, ax_plot_1)
    #plot_pitch_trajectories_with_endpoints_3d(filtered_data, ax_plot_2)
    plotly_test.plot_pitch_trajectories_with_endpoints_3d(filtered_data)
    # Load and display the cropped image
    img = mpimg.imread('cropped_3d_plot.png')
    ax_plot_2.imshow(img)
    ax_plot_2.axis('off')  # Hide axis for the image plot

    # Add footer text
    ax_footer.text(0, 1, 'By: Garrett Ramos', ha='left', va='top', fontsize=24)
    #ax_footer.text(0.5, 1, 'Colour Coding Compares to League Average By Pitch', ha='center', va='top', fontsize=16)
    ax_footer.text(1, 1, 'Data: MLB, Fangraphs\nImages: MLB, ESPN', ha='right', va='top', fontsize=24)

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Show the figure
    plt.show()