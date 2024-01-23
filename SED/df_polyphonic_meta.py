# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 23:12:57 2024

@author: Snehit
"""

#Structure
'''
1. Imports
2. Dictionary location: User input
3. df: df of given STeLin-US metadata file
4. df1: df for splitting any events longer than 5 sec duration into separate clips of 5sec
5. df2: df of removed false event files and merged overlapping events into one group
'''

# Import necessary packages
import pandas as pd
import os
import json
import librosa
import numpy as np
from tqdm import tqdm

# User Define: Dictionary location of STeLin-US Dataset
Dir = "E:\BIIC\Dataset\Python\Data Sort\Mix\Output"

Audio_path = Dir+"\STeLiN-US\Audio"


# Raw metadata from given STeLin-US
df = pd.read_excel(Dir+'\STeLiN-US\MetaData.xlsx')
print('\ndf: \n', df[:5])           # print fist 5 rows of given metadata

Events = list(df.columns.values[1::2])  # All Events list from metadata


## def Clip_TrimTo_5Sec: From one file of 5 min (300 sec) to Clips of 5 sec or less containing Events
##     Note: clips of car_horn and dog_bark are also less than 2 sec and vehicle is of 2 sec this all fall below 5 sec lim
def Clip_TrimTo_5Sec(df, Events):
    dict_Events = {'Wavefile': [],
                   'Events': [],
                   'Clip': []}
    for A_clip in df['Wavefile']:
      for event in Events:
        if df[event][df['Wavefile'] == A_clip].values[0] != 0:
          Event_Time_list = json.loads(df[event + ' [Tstart Tend]'][df['Wavefile']==A_clip].values[0])
    
          for Time in Event_Time_list:
            if Time[1]-Time[0]>5000:           # 5000 is in milliseconds (i.e. 5sec)
              Duration = Time[1] - Time[0]
              T0 = Time[0]
              for Splits in range(int(np.ceil(Duration/5000))):
                T1 = min(Time[1], T0+5000)
                T_split = [T0, T1]
                T0 += 5000
    
                dict_Events['Wavefile'].append(A_clip)
                dict_Events['Events'].append(event)
                dict_Events['Clip'].append(tuple(T_split))
                
            else:
              dict_Events['Wavefile'].append(A_clip)
              dict_Events['Events'].append(event)
              dict_Events['Clip'].append(tuple(Time))
             
    df1 = pd.DataFrame(dict_Events).sort_values(["Wavefile", "Clip"])
    df1.reset_index(inplace=True, drop=True)
    return df1
  
    

## def Remove0Energy_MergePolyphonicClips: Removing False Event clips and merging events fall in same clip
def Remove0Energy_MergePolyphonicClips(df, df1, Audio_path, STD, EventsInStr):
    dict_Events = {'Wavefile': [],
                   'Events': [],
                   'Clip': []}
    All_clip = list(df['Wavefile'].sort_values())     # list of All audio filenames
    
    # Scanning through Each Audio clip one by one
    for one_clip in tqdm(All_clip):
        y, sr = librosa.load(os.path.join(Audio_path,one_clip.split("_")[1], one_clip))
        tempDF = df1[df1['Wavefile']==one_clip]   # temporary df for each Audio file
        
        energy_centre_pt = []
        Event_list = []
        
        # Scan through one audio file: to group each event under overlapping duration
        for index, row in tempDF.iterrows():
            
            s_pt = [int(row['Clip'][0]*sr/1000), int(row['Clip'][1]*sr/1000)]
            Segment = y[s_pt[0]:s_pt[1]]**2
            Energy = sum(abs(Segment**2))
            
            # Eliminating false Event segments: if energy in the clip is zero then it's a false event clip
            if Energy!=0:
                centre_pt = sum(np.arange(s_pt[0], s_pt[1]) * Segment**2) / sum(Segment**2)
                energy_centre_pt.append(centre_pt)
                
                Event_list.append(row['Events'])
            
            # Grouping if the energy centres of the clips fall are under 1 sec of standard deviation
            if np.std(energy_centre_pt)/sr > STD:  # IMPORTANT: if standard deviation amongs the centers is within STD (default=1sec) group them in one clip
                
                ## 1. -- dict add: wavefile -- ##
                dict_Events['Wavefile'].append(row['Wavefile'])
                
                ## 2. -- dict add: Event or Events -- ##
                Event_list = list(set(Event_list[:-1]))
                Event_list.sort()
                
                if EventsInStr==True:
                    Event_list = str(Event_list)
                    Event_list = Event_list.replace('[', '').replace(']', '').replace("'", '')
                   
                dict_Events['Events'].append(Event_list)
                
                ## 3. -- dict add: Clip duration -- ##
                if len(energy_centre_pt[:-1])==0:
                    Time = [np.array([temp_s_pt[0]*1000/sr, temp_s_pt[1]*1000/sr], dtype=int)]
                else: 
                    if np.mean(energy_centre_pt[:-1])*1000/sr - 2500 < 0:
                        Time = list(np.array(0 + np.array([0, 5000]), dtype=int))
                    elif np.mean(energy_centre_pt[:-1])*1000/sr + 2500 > 5*60*1000:
                        Time = list(np.array(5*60*1000 + np.array([-5000, 0]), dtype=int))
                    else:
                        Time = list(np.array(np.mean(energy_centre_pt[:-1])*1000/sr + np.array([-2500, 2500]), dtype=int))
                
                dict_Events['Clip'].append(tuple(Time))
                
                # Cache
                Event_list = [row['Events']]
                energy_centre_pt = [energy_centre_pt[-1]]
                temp_s_pt = s_pt
                
                ## Correction: Last clip in the file: if doesn't merge to any group then creat it as seperate 
                if index == tempDF.index[-1]:
                    ## 1. -- dict add: wavefile -- ##
                    dict_Events['Wavefile'].append(row['Wavefile'])
                    
                    ## 2. -- dict add: Event -- ##
                    if EventsInStr==False:
                        if type(row['Events']!=list) :
                            row['Events'] = [row['Events']]
                    dict_Events['Events'].append(row['Events'])
                    
                    ## 3. -- dict add: Clip duration -- ##
                    dict_Events['Clip'].append(tuple(row['Clip']))
                    
    df2 = pd.DataFrame(dict_Events)  ## final convert dict to dataframe
    return df2

## DF for 5 Sec or less clips
df1 = Clip_TrimTo_5Sec(df, Events)
print('\nDF1: \n', df1[0:5]) # print fist 5 rows in df of clipped 5sec files

## DF of 5 Sec (fixed) with multiple events merged STD set for 1sec (Default: lower value is better)
df2 = Remove0Energy_MergePolyphonicClips(df, df1, Audio_path, STD=1, EventsInStr=False)
print('\nDF1', df2[0:5]) # print fist 5 rows in final merged df


# Save df in excel file
df2.to_excel("df_polyphonic_meta.xlsx")

# Save in pickel file: retains the dtype
df2.to_pickle("df_polyphonic_meta.pkl")
