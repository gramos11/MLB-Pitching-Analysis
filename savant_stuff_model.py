# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 19:47:42 2024

@author: Graduate
"""
#from pybaseball import statcast
#from pybaseball import cache
import pandas as pd
import numpy as np
from math import pi


#cache.enable()

Data = pd.read_csv('py_pitch_data_2020_2024.csv')
#columns_to_check = [
#    'release_speed', 'release_spin_rate', 'release_extension', 'az', 'ax'
#]
#data_clean = Data.dropna(subset=columns_to_check)

#data_clean['api_break_z_with_gravity'] = data_clean['api_break_z_with_gravity'] * 12
#data_clean['pfx_x'] = data_clean['pfx_x'] * 12
#data_clean['pfx_z'] = data_clean['pfx_z'] * 12
#data_clean['api_break_x_batter_in'] = data_clean['api_break_x_batter_in'] * 12
#data_clean['api_break_x_arm'] = data_clean['api_break_x_arm'] * 12

#data_clean['break_diff'] = (523 / data_clean['release_speed'])**2
#data_clean['ivb'] = data_clean['api_break_z_with_gravity'] - data_clean['break_diff']
#data_clean['ivb'] = data_clean['ivb'] * -1

def release_angles(df):
    vyo = df['vy0']
    ay = df['ay']
    y_release = df['release_pos_y']
    value = ((vyo**2) - 2 * ay * (50 - y_release))**0.5
    df['tr'] = (-vyo - value)/ay
    tr = df['tr']
    
    df['vyr'] = vyo + ay * tr
    vyr = df['vyr']
    
    vxo = df['vx0']
    ax = df['ax']
    df['vxr'] = vxo + ax * tr
    vxr = df['vxr']
    
    vzo = df['vz0']
    az = df['az']
    df['vzr'] = vzo + az * tr
    vzr = df['vzr']
    
    vra = -np.degrees(np.arctan(vzr / np.sqrt(vyr**2+vxr**2)))
    hra = -np.degrees(np.arctan(vxr / np.sqrt(vyr**2+vxr**2)))
    
    haa = -np.degrees(np.arctan((vxr/value) * (180/pi)))
    vaa = -np.degrees(np.arctan((vzr/value) * (180/pi)))
    
    df['vra'] = vra
    df['hra'] = hra
    df['haa'] = haa
    df['vaa'] = vaa
    
    return df

Data = release_angles(Data)

def cleanData(df):
    events_to_replace = [
            'force_out', 'double_play', 'sac_fly', 'grounded_into_double_play', 
            'fielders_choice_out', 'sac_bunt', 'fielders_choice', 
            'sac_fly_double_play', 'triple_play'
        ]
    df['events'] = df['events'].replace(events_to_replace, 'field_out')

    events_to_replace = [
            'foul_bunt', 'foul_pitchout']
    df['description'] = df['description'].replace(events_to_replace, 'foul')        

    events_to_replace = [
            'blocked_ball', 'pitchout']
    df['description'] = df['description'].replace(events_to_replace, 'ball')        

    events_to_replace = [
            'swinging_strike_blocked', 'missed_bunt', 'foul_tip', 'bunt_foul_tip']
    df['description'] = df['description'].replace(events_to_replace, 'swinging_strike')
            
    allowed_events = ['field_out', 'single', 'home_run', 'double', 'triple', 'strikeout', 'hit_by_pitch', 'walk']
    df = df[~((df['description'] == 'hit_into_play') & (~df['events'].isin(allowed_events)))]        

    df['outcome'] = np.where(
        df['events'].isin(['hit_by_pitch', 'strikeout', 'walk']),  # First condition
        df['events'],
        np.where(
            df['description'].isin(['swinging_strike', 'ball', 'foul', 'called_strike']),  # Second condition
            df['description'],
            np.where(
                df['description'] == 'hit_into_play',  # Third condition
                df['events'],
                None  # Default for other cases
            )
        )
    )

    df = df[~df['events'].isin(['catcher_interf', 'strikeout_double_play', 'truncated_pa'])]

    targets = pd.read_csv('run_values.csv')
    #targets = df.groupby(['balls', 'strikes', 'outcome'])['delta_run_exp'].mean().reset_index()
    targets.rename(columns={'delta_run_exp': 'avg_delta_run_exp'}, inplace=True)
    targets.rename(columns={'event': 'outcome'}, inplace=True)

        
    #targets.drop(targets[targets['balls'] == 4].index, inplace=True)
    #targets.drop(targets[targets['strikes'] == 3].index, inplace=True)

    df = df.merge(
        targets[['balls', 'strikes', 'outcome', 'avg_delta_run_exp']], 
        on=['balls', 'strikes', 'outcome'], 
        how='left'
    )

    df.rename(columns={'avg_delta_run_exp': 'target'}, inplace=True)

    df.drop(df[df['balls'] == 4].index, inplace=True)
    df.drop(df[df['strikes'] == 3].index, inplace=True)
    
    return df

Data = cleanData(Data)

import polars as pl
Data = pl.from_pandas(Data)

def feature_engineering(df: pl.DataFrame) -> pl.DataFrame:
    # Extract the year from the game_date column
    df = df.with_columns(
        pl.col('game_date').str.slice(0, 4).alias('game_year')
    )

    # Mirror horizontal break for left-handed pitchers
    df = df.with_columns(
        pl.when(pl.col('p_throws') == 'L')
        .then(-pl.col('ax'))
        .otherwise(pl.col('ax'))
        .alias('ax')
    )

    # Mirror horizontal release point for left-handed pitchers
    df = df.with_columns(
        pl.when(pl.col('p_throws') == 'L')
        .then(pl.col('release_pos_x'))
        .otherwise(-pl.col('release_pos_x'))
        .alias('release_pos_x')
    )

    # Define the pitch types to be considered
    pitch_types = ['SI', 'FF', 'FC']

    # Filter the DataFrame to include only the specified pitch types
    df_filtered = df.filter(pl.col('pitch_type').is_in(pitch_types))

    # Group by pitcher_id and year, then aggregate to calculate average speed and usage percentage
    df_agg = df_filtered.group_by(['pitcher', 'game_year', 'pitch_type']).agg([
        pl.col('release_speed').mean().alias('avg_fastball_speed'),
        pl.col('az').mean().alias('avg_fastball_az'),
        pl.col('ax').mean().alias('avg_fastball_ax'),
        pl.len().alias('count')
    ])

    # Sort the aggregated data by count and average fastball speed
    df_agg = df_agg.sort(['count', 'avg_fastball_speed'], descending=[True, True])
    df_agg = df_agg.unique(subset=['pitcher', 'game_year'], keep='first')

    # Join the aggregated data with the main DataFrame
    df = df.join(df_agg, on=['pitcher', 'game_year'])

    # If no fastball, use the fastest pitch for avg_fastball_speed
    df = df.with_columns(
        pl.when(pl.col('avg_fastball_speed').is_null())
        .then(pl.col('release_speed').max().over('pitcher'))
        .otherwise(pl.col('avg_fastball_speed'))
        .alias('avg_fastball_speed')
    )

    # If no fastball, use the fastest pitch for avg_fastball_az
    df = df.with_columns(
        pl.when(pl.col('avg_fastball_az').is_null())
        .then(pl.col('az').max().over('pitcher'))
        .otherwise(pl.col('avg_fastball_az'))
        .alias('avg_fastball_az')
    )

    # If no fastball, use the fastest pitch for avg_fastball_ax
    df = df.with_columns(
        pl.when(pl.col('avg_fastball_ax').is_null())
        .then(pl.col('ax').max().over('ax'))
        .otherwise(pl.col('avg_fastball_ax'))
        .alias('avg_fastball_ax')
    )

    # Calculate pitch differentials
    df = df.with_columns(
        (pl.col('release_speed') - pl.col('avg_fastball_speed')).alias('speed_diff'),
        (pl.col('az') - pl.col('avg_fastball_az')).alias('az_diff'),
        (pl.col('ax') - pl.col('avg_fastball_ax')).abs().alias('ax_diff')
    )

    # Cast the year column to integer type
    df = df.with_columns(
        pl.col('game_year').cast(pl.Int64)
    )

    return df

Data = feature_engineering(Data.clone())

Data = Data.to_pandas()
train = Data[Data['game_year'].isin([2020, 2021, 2022, 2023])]
train = train.dropna(subset=['target'])
#test = Data[Data['game_year'].isin([2023])]
#test = test.dropna(subset=['target'])
test_2024 = Data[Data['game_year'].isin([2024])]
test_2024 = test_2024.dropna(subset=['target'])

X_train = train[['release_speed', 'release_spin_rate', 'speed_diff', 'ax_diff', 'az_diff', 'ax', 'az', 'release_extension', 'release_pos_x', 'release_pos_z', 'hra', 'vra']].astype(np.float32)
y_train = train['target'].astype(np.float32)

X_train = X_train.apply(pd.to_numeric, errors='coerce')
y_train = y_train.apply(pd.to_numeric, errors='coerce')

#X_test = test[['release_speed', 'release_spin_rate', 'speed_diff', 'ax_diff', 'az_diff', 'ax', 'az', 'release_extension', 'release_pos_x', 'release_pos_z', 'hra', 'vra']].astype(np.float32)
#y_test = test['target'].astype(np.float32)

#X_test = X_test.apply(pd.to_numeric, errors='coerce')
#y_test = y_test.apply(pd.to_numeric, errors='coerce')

X_test_2024 = test_2024[['release_speed', 'release_spin_rate', 'speed_diff', 'ax_diff', 'az_diff', 'ax', 'az', 'release_extension', 'release_pos_x', 'release_pos_z', 'hra', 'vra']].astype(np.float32)
y_test_2024 = test_2024['target'].astype(np.float32)

X_test_2024 = X_test_2024.apply(pd.to_numeric, errors='coerce')
y_test_2024 = y_test_2024.apply(pd.to_numeric, errors='coerce')


from sklearn.metrics import mean_squared_error, r2_score
from lightgbm import LGBMRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

model = make_pipeline(
    RobustScaler(),            # Robust Scaler to scale the features
    LGBMRegressor(
        n_estimators=1000,         # Number of boosting rounds (trees) to be built.
        learning_rate=0.01,        # Step size shrinkage used to prevent overfitting. Smaller values require more boosting rounds.
        num_leaves=31,             # Maximum number of leaves in one tree. Controls the complexity of the model.
        max_depth=-1,              # Maximum depth of the tree. -1 means no limit.
        min_child_samples=20,      # Minimum number of data points required in a leaf. Helps control overfitting.
        subsample=0.8,             # Fraction of data to be used for each boosting round. Helps prevent overfitting.
        colsample_bytree=0.8,      # Fraction of features to be used for each boosting round. Helps prevent overfitting.
        reg_alpha=0.1,             # L1 regularization term on weights. Helps prevent overfitting.
        reg_lambda=0.2,            # L2 regularization term on weights. Helps prevent overfitting.
        random_state=42,           # Seed for reproducibility.
        force_row_wise=True        # Force row-wise (data parallel) computation. Useful for handling large datasets.
    )
)

# Fit the model to the training data
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test_2024)
test_2024['pred'] = y_pred

# Step 5: Evaluate the model
mse = mean_squared_error(y_test_2024, y_pred)
r2 = r2_score(y_test_2024, y_pred)
rmse = np.sqrt(mse)

print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared: {r2:.4f}")

import matplotlib.pyplot as plt
import seaborn as sns  
sns.set_theme(style='whitegrid')

lgbm_model = model.named_steps['lgbmregressor']

# Extract feature importances
feature_importances = lgbm_model.feature_importances_

features = X_train.columns
# Assuming 'features' is a list of feature names
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances
})

# Sort the DataFrame by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importances')
plt.gca().invert_yaxis()
plt.show()


target_mean = test_2024['pred'].mean()
target_std = test_2024['pred'].std()

# Standardize the target column to create a z-score for 2023
test_2024['target_zscore'] = (test_2024['pred'] - target_mean) / target_std


# Convert the z-score to tj_stuff_plus for 2023
test_2024['tj_stuff_plus'] = 100 - (test_2024['target_zscore'] * 10)


# Aggregate tj_stuff_plus by pitcher_id and year for 2023
test_2024_agg = test_2024.groupby(['player_name']).agg(
    count=('tj_stuff_plus', 'size'),           # Count the number of occurrences
    mean=('tj_stuff_plus', 'mean')            # Calculate the mean
).reset_index()


pitch_colours = {
    ## Fastballs ##
    'FF': {'colour': '#FF007D', 'name': '4-Seam Fastball'},
    'FA': {'colour': '#FF007D', 'name': 'Fastball'},
    'SI': {'colour': '#98165D', 'name': 'Sinker'},
    'FC': {'colour': '#BE5FA0', 'name': 'Cutter'},

    ## Offspeed ##
    'CH': {'colour': '#F79E70', 'name': 'Changeup'},
    'FS': {'colour': '#FE6100', 'name': 'Splitter'},
    'SC': {'colour': '#F08223', 'name': 'Screwball'},
    'FO': {'colour': '#FFB000', 'name': 'Forkball'},

    ## Sliders ##
    'SL': {'colour': '#67E18D', 'name': 'Slider'},
    'ST': {'colour': '#1BB999', 'name': 'Sweeper'},
    'SV': {'colour': '#376748', 'name': 'Slurve'},

    ## Curveballs ##
    'KC': {'colour': '#311D8B', 'name': 'Knuckle Curve'},
    'CU': {'colour': '#3025CE', 'name': 'Curveball'},
    'CS': {'colour': '#274BFC', 'name': 'Slow Curve'},
    'EP': {'colour': '#648FFF', 'name': 'Eephus'},

    ## Others ##
    'KN': {'colour': '#867A08', 'name': 'Knuckleball'},
    'PO': {'colour': '#472C30', 'name': 'Pitch Out'},
    'UN': {'colour': '#9C8975', 'name': 'Unknown'},
}

dict_colour = dict(zip(pitch_colours.keys(), [pitch_colours[key]['colour'] for key in pitch_colours]))
# Create a dictionary mapping pitch types to their colors
dict_pitch = dict(zip(pitch_colours.keys(), [pitch_colours[key]['name'] for key in pitch_colours]))

# Create a dictionary mapping pitch types to their colors
dict_pitch_desc_type = dict(zip([pitch_colours[key]['name'] for key in pitch_colours],pitch_colours.keys()))


# Create a dictionary mapping pitch types to their colors
dict_pitch_name = dict(zip([pitch_colours[key]['name'] for key in pitch_colours], 
                           [pitch_colours[key]['colour'] for key in pitch_colours]))

import matplotlib.pyplot as plt
import seaborn as sns

# Create subplots for the histograms
fig, ax = plt.subplots(2, 1, figsize=(10, 10))

# Plot the histogram of tj_stuff_plus for specific pitch types
sns.histplot(data=test_2024[test_2024['pitch_type'].isin(['FF', 'SI', 'FC', 'SL', 'ST', 'CH', 'FS', 'CU', 'KC'])], 
             x='tj_stuff_plus', 
             binrange=[60, 140], 
             bins=40,
             ax=ax[0]
             )

# Set the title of the first subplot
ax[0].set_title('2024 Pitch Stuff+ Distribution')

# Plot the histogram of tj_stuff_plus for specific pitch types, colored by pitch type
sns.histplot(data=test_2024[test_2024['pitch_type'].isin(['FF', 'SI', 'FC', 'SL', 'ST', 'CH', 'FS', 'CU', 'KC'])], 
             x='tj_stuff_plus', 
             binrange=[60, 140], 
             bins=40,
             hue='pitch_type',
             multiple='stack',  
             palette=dict_colour,
             ax=ax[1]
             )

# Set the title of the second subplot
ax[1].set_title('2024 Pitch Stuff+ Distribution by Pitch Type')

# Set the x-axis label
ax[0].set_xlabel('tjStuff+')
ax[1].set_xlabel('tjStuff+')

# Change the legend title to 'Pitch Type'
ax[1].get_legend().set_title("Pitch Type")   

# Adjust layout to prevent overlap
fig.tight_layout()


import seaborn as sns
import matplotlib.pyplot as plt

# Calculate the median of tj_stuff_plus for each pitch_type
mean_values = test_2024[test_2024['pitch_type'].isin(['FF', 'SI', 'FC', 'SL', 'ST', 
                                                        'CH', 'FS', 'CU', 'KC'])].groupby('pitch_type')['tj_stuff_plus'].median().sort_values(ascending=False)

# Map the median values to the dataframe
test_2024['tj_stuff_plus_mean'] = test_2024['pitch_type'].map(mean_values.to_dict())

# Sort the dataframe by the median values of tj_stuff_plus
test_2024 = test_2024.sort_values(by='tj_stuff_plus_mean', ascending=False)

# Create a subplot for the boxen plot
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

# Plot the boxen plot of tj_stuff_plus for specific pitch types, colored by pitch type
bp = sns.boxenplot(data=test_2024[test_2024['pitch_type'].isin(['FF', 'SI', 'FC', 'SL', 'ST', 'CH', 'FS', 'CU', 'KC'])], 
               x='tj_stuff_plus', 
               y='pitch_type',
               palette=dict_colour,
               ax=ax,
               showfliers=False,  # Do not show outliers
               k_depth=6          # Number of boxes to draw
               )

bp.set_yticklabels([dict_pitch[x.get_text()] + f' ({x.get_text()})' for x in bp.get_yticklabels()])

# Annotate the median values on the plot
for index, row in mean_values.reset_index().iterrows():
    ax.text(row['tj_stuff_plus'], 
            index, 
            f'{row["tj_stuff_plus"]:.0f}', 
            color='black', 
            ha="center", 
            va="center",
            bbox=dict(facecolor='white', alpha=1,edgecolor='k')  # White background for the text
            )


# Set the x-axis limits
ax.set_xlim(60, 140)

# Set the title of the plot
ax.set_title('2024 tjStuff+ Distribution and Median by Pitch Type')

# Set the x-axis and y-axis label
ax.set_xlabel('tjStuff+')
ax.set_ylabel('Pitch Type')

# Display the plot
plt.show()


best_pitches_2024 = (
    test_2024.groupby(['pitch_type', 'player_name'])
      .agg(
          avg_tj_stuff_plus=('tj_stuff_plus', 'mean'),  # Calculate average
          count=('tj_stuff_plus', 'size')              # Count occurrences
      )
      .reset_index()  # Reset index to make it a flat DataFrame
)













