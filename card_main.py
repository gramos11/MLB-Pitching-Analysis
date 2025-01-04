# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 16:05:16 2024

@author: Graduate
"""

import matplotlib as mpl
import card_funcs

mpl.rcParams['figure.dpi'] = 300

key, table2, table3, stuff_data, player_data, filtered_data, stats = card_funcs.get_data(player_name='emmanuel clase')

#card_funcs.plot_pitch_break(table3.copy(), table2.copy(), plt.subplots(figsize=(6, 6))[1])
#card_funcs.plot_stuff_plus(player_data.copy(), stuff_data.copy(), plt.subplots(figsize=(10, 6))[1])
#card_funcs.plot_pitch_location(filtered_data.copy(), plt.subplots(figsize=(8, 10))[1])
#card_funcs.plot_pitch_trajectories_with_endpoints_3d(filtered_data.copy(), ax=None)
#card_funcs.player_headshot(pitcher_id=key, ax=plt.subplots(figsize=(1, 1))[1])
#card_funcs.player_bio(key, ax=plt.subplots(figsize=(20, 4))[1])
#card_funcs.plot_logo(key, image_dict, ax=plt.subplots(figsize=(1, 1))[1])
#df_fangraphs = fangraphs_pitching_leaderboards(season = 2024)
#stats = ['IP','TBF','WHIP','ERA', 'FIP', 'K%', 'BB%', 'K-BB%']
#card_funcs.fangraphs_pitcher_stats(key, plt.subplots(figsize=(10, 1))[1], stats, 2024)

card_funcs.pitching_dashboard(key, table3.copy(), table2.copy(), filtered_data.copy(), player_data.copy(), stuff_data.copy(), stats.copy())
