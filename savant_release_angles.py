# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 18:19:20 2024

@author: Graduate
"""


import numpy as np


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
    
    vra = -np.degrees(np.arctan(vzr / vyr))
    hra = -np.degrees(np.arctan(vxr / vyr))
    
    df['vra'] = vra
    df['hra'] = hra
    
    return df

