# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 16:20:41 2024

@author: Graduate
"""
import pandas as pd
from pybaseball import  playerid_lookup
from pybaseball import  statcast_pitcher
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Ellipse
from matplotlib.font_manager import FontProperties
from matplotlib.patheffects import withStroke


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

# Streamlit interface
st.title('Pitch Break Visualization')
st.write("This app visualizes pitch breaks for pitchers during the 2024 MLB season.")

# User input for player name (Sean Manaea) or another player
player_name = st.text_input('Enter player name')

if st.button("Load Data"):
    try:
        name_list = player_name.lower().split()
        first_name = name_list[0]
        last_name = name_list[1]
        table = playerid_lookup(last_name, first_name)
        key = table['key_mlbam'][0]
        table2 = statcast_pitcher('2024-03-28', '2024-11-01', key)
        table2 = table2.dropna(subset=["pfx_x", "pfx_z", "p_throws"])
        if table2["p_throws"].iloc[0] == "L":
            table2["pfx_x"] = -table2["pfx_x"]
        
        #table2['break_diff'] = (523 / table2['release_speed'])**2
        #table2['ivb'] = table2['api_break_z_with_gravity'] - table2['break_diff']
        #table2['ivb'] = table2['ivb'] * -1
        #table2['hb'] = table2['api_break_x_arm'] * 12
        table2["pfx_x"] = table2["pfx_x"] * 12
        table2["pfx_z"] = table2["pfx_z"] * 12
        table3 = table2.groupby(["pitch_type", "game_date"])[["pfx_x", "pfx_z"]].mean().reset_index()
        
        # Initialize the plot
        fig1, ax1 = plt.subplots(figsize=(10, 6))

        # Assign colors to each pitch_type using seaborn's color palette
        palette = sns.color_palette("husl", len(table3["pitch_type"].unique()))
        color_mapping = dict(zip(table3["pitch_type"].unique(), palette))

        # Plot scatter points and ellipses for each pitch_type
        for pitch_type, group in table3.groupby("pitch_type"):
            ax1.scatter(group["pfx_x"], group["pfx_z"], label=pitch_type, 
                      color=color_mapping[pitch_type], alpha=0.7, edgecolor='black', s=100)
        for pitch_type, group in table2.groupby("pitch_type"):
            plot_confidence_ellipse(group["pfx_x"], group["pfx_z"], ax1, 
                                    color=color_mapping[pitch_type], ellipse_alpha=0.3)

        # Set axis limits
        ax1.set_xlim(-25, 25)
        ax1.set_ylim(-25, 25)
        
        ax1.axhline(0, color='orange', linestyle='--', linewidth=1)  # Horizontal line (y=0)
        ax1.axvline(0, color='orange', linestyle='--', linewidth=1)

        # Add labels and title
        ax1.set_xlabel("Horizontal Break (inches)")
        ax1.set_ylabel("Induced Vertical Break (inches)")
        ax1.set_title(f"Pitch Break - {first_name.capitalize()} {last_name.capitalize()}")

        # Show grid and legend
        ax1.grid(True)
        ax1.legend(title="Pitch Type")

        # Display the plot in Streamlit
        st.pyplot(fig1)
        
        
        
        stuff_data = pd.read_csv("best_pitches_2024.csv")
        
        # Ensure that the 'player_name' column is correctly formatted (e.g., "First Last")
        stuff_data['player_name'] = stuff_data['player_name'].apply(lambda x: ' '.join(x.split(', ')[::-1]))  # Convert "Last, First" to "First Last"
        
        # Filter the data for the specified player
        player_data = stuff_data[stuff_data['player_name'] == f"{first_name.capitalize()} {last_name.capitalize()}"]

        if player_data.empty:
            st.error(f"No data found for {first_name.capitalize()} {last_name.capitalize()} in the CSV file.")
        else:
            # Get unique pitch types from the player's data
            pitch_types = player_data['pitch_type'].unique()

            # Create subplots for each pitch type
            n_pitches = len(pitch_types)
            fig2, axes = plt.subplots(1, n_pitches, figsize=(5 * n_pitches, 6), sharey=True)
            
            if n_pitches == 1:
                axes = [axes]  # Ensure axes is iterable when there's only one pitch type

            # Assign colors to each pitch type using seaborn's color palette
            palette = sns.color_palette("husl", n_colors=len(pitch_types))

            # Plot for each pitch type
            for ax, pitch_type, color in zip(axes, pitch_types, palette):
                # Filter data for the pitch type
                pitch_data = stuff_data[stuff_data['pitch_type'] == pitch_type]
                player_pitch = player_data[player_data['pitch_type'] == pitch_type]
                
                # Create swarm plot with the color from the palette and a lower alpha for translucency
                sns.swarmplot(
                    data=pitch_data, 
                    x="pitch_type", 
                    y="avg_tj_stuff_plus", 
                    color=color, 
                    size=5, 
                    alpha=0.3,  # Increased translucency of data points
                    ax=ax
                )
                
                # Highlight the player's pitch grade with more opaque color
                for _, row in player_pitch.iterrows():
                    ax.scatter(
                        row['pitch_type'], row['avg_tj_stuff_plus'], 
                        color=color, s=800, edgecolor='black', alpha=0.8,  # More opaque for the player's dot
                        zorder=5
                    )
                    # Dynamically adjust font size and place text inside the dot
                    pitch_grade = round(row['avg_tj_stuff_plus'])
                    
                    # Apply font properties using fontdict
                    font_properties = FontProperties(
                                family='Arial',
                                weight='heavy',
                                size=12
                            )
                    # Add annotation with rounded pitch grade, inside the dot
                    ax.text(
                        row['pitch_type'], row['avg_tj_stuff_plus'], 
                        f"{pitch_grade}",  # Round pitch grade to nearest whole number
                        color='white', ha='center', va='center', fontproperties=font_properties, path_effects=[withStroke(linewidth=2, foreground='black')], zorder=10
                    )
                
                # Add horizontal line at 100 (midpoint)
                ax.axhline(100, color='gray', linestyle='--', linewidth=1)
                
                # Set subplot title and limits
                ax.set_title(f"{first_name.capitalize()} {last_name.capitalize()} - {pitch_type}", fontsize=12)
                ax.set_ylim(70, 130)
                ax.set_xlabel("")  # Remove x-axis label
                ax.set_ylabel("Stuff+", fontsize=12)

            # Add overall figure title
            fig2.suptitle(f"{first_name.capitalize()} {last_name.capitalize()}'s Pitch Arsenal - Stuff+ Comparison", fontsize=16, y=1.02)
            plt.tight_layout()

            # Display the second plot (Stuff+ Comparison from CSV)
            st.pyplot(fig2)
        
    except Exception as e:
        st.error(f"Error loading data: {e}")

