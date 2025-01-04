# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 20:20:26 2024

@author: Graduate
"""
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
from matplotlib.patheffects import withStroke


def reformat_name(name):
    last, first = name.split(', ')
    return f"{first} {last}"

# Apply the reformatting function to the 'player_name' column
best_pitches_2024['formatted_player_name'] = best_pitches_2024['player_name'].apply(reformat_name)

# Specify the player
specific_player = "Clase, Emmanuel"
formatted_player = reformat_name(specific_player)

# Get all unique pitch types for the player
player_data = best_pitches_2024[best_pitches_2024['player_name'] == specific_player]
pitch_types = player_data['pitch_type'].unique()

# Use Seaborn's "husl" palette to generate distinct colors
palette = sns.color_palette("husl", n_colors=len(pitch_types))

# Create subplots for each pitch type
n_pitches = len(pitch_types)
fig, axes = plt.subplots(1, n_pitches, figsize=(5 * n_pitches, 6), sharey=True)

# If only 1 pitch, make axes iterable
if n_pitches == 1:
    axes = [axes]

# Plot for each pitch type
for ax, pitch_type, color in zip(axes, pitch_types, palette):
    # Filter data for the pitch type
    pitch_data = best_pitches_2024[best_pitches_2024['pitch_type'] == pitch_type]
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
                    weight=1000,
                    size=12
                )
        # Add annotation with rounded pitch grade, inside the dot
        ax.text(
            row['pitch_type'], row['avg_tj_stuff_plus'], 
            f"{pitch_grade}",  # Round pitch grade to nearest whole number
            color='white', ha='center', va='center', fontproperties=font_properties,path_effects=[withStroke(linewidth=3, foreground='black')], zorder=10
        )
    
    # Add horizontal line at 100 (midpoint)
    ax.axhline(100, color='gray', linestyle='--', linewidth=1)
    
    # Set subplot title and limits
    ax.set_title(f"{formatted_player} - {pitch_type}", fontsize=12)
    ax.set_ylim(70, 130)
    ax.set_xlabel("")  # Remove x-axis label
    ax.set_ylabel("Stuff+", fontsize=12)

# Add overall figure title
fig.suptitle(f"{formatted_player}'s Pitch Arsenal - Stuff+ Comparison", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()