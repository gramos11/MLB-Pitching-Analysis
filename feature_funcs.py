# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 15:57:56 2024

@author: Graduate
"""
def feature_engineering(clean_data):
    # Extract the year from game_date
    clean_data['game_year'] = clean_data['game_date'].str[:4].astype(int)
    
    # Mirror horizontal break for left-handed pitchers
    clean_data.loc[clean_data['p_throws'] == 'L', 'ax'] *= -1
    
    # Mirror horizontal release point for left-handed pitchers
    clean_data.loc[clean_data['p_throws'] != 'L', 'release_pos_x'] *= -1

    # Filter for relevant pitch types
    filtered_data = clean_data[clean_data['pitch_type'].isin(['FF', 'SI', 'FC'])]
    pitch_counts = filtered_data.groupby(['pitcher', 'game_year', 'pitch_type']).size().reset_index(name='count')
    pitch_counts = pitch_counts.merge(
        filtered_data.groupby(['pitcher', 'game_year', 'pitch_type'])['release_speed'].mean().reset_index(),
        on=['pitcher', 'game_year', 'pitch_type']
    )

    # Determine primary pitch by count and speed
    primary_pitch_mapping = (
        pitch_counts.sort_values(['pitcher', 'game_year', 'count', 'release_speed'], ascending=[True, True, False, False])
        .drop_duplicates(subset=['pitcher', 'game_year'])
        .set_index(['pitcher', 'game_year'])['pitch_type']
        .to_dict()
    )
    clean_data['pitcher_primary_pitch'] = clean_data.set_index(['pitcher', 'game_year']).index.map(primary_pitch_mapping)

    # Calculate average speed, ax, and az for primary pitch
    clean_data['primary_pitch_velo'] = clean_data.set_index(['pitcher', 'game_year']).index.map(
        pitch_counts.set_index(['pitcher', 'game_year', 'pitch_type'])['release_speed'].to_dict()
    )
    clean_data['primary_pitch_ax'] = clean_data.set_index(['pitcher', 'game_year']).index.map(
        pitch_counts.set_index(['pitcher', 'game_year', 'pitch_type'])['ax'].to_dict()
    )
    clean_data['primary_pitch_az'] = clean_data.set_index(['pitcher', 'game_year']).index.map(
        pitch_counts.set_index(['pitcher', 'game_year', 'pitch_type'])['az'].to_dict()
    )

    # Handle missing fastball data
    clean_data['primary_pitch_velo'] = clean_data['primary_pitch_velo'].fillna(
        clean_data.groupby('pitcher')['release_speed'].transform('max')
    )
    clean_data['primary_pitch_ax'] = clean_data['primary_pitch_ax'].fillna(
        clean_data.groupby('pitcher')['ax'].transform('max')
    )
    clean_data['primary_pitch_az'] = clean_data['primary_pitch_az'].fillna(
        clean_data.groupby('pitcher')['az'].transform('max')
    )

    # Calculate pitch differentials
    clean_data['speed_diff'] = clean_data['release_speed'] - clean_data['primary_pitch_velo']
    clean_data['ax_diff'] = (clean_data['ax'] - clean_data['primary_pitch_ax']).abs()
    clean_data['az_diff'] = clean_data['az'] - clean_data['primary_pitch_az']

    # Explicit casting of game_year
    clean_data['game_year'] = clean_data['game_year'].astype(int)

    return clean_data
