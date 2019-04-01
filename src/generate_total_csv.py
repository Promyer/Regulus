#!/usr/bin/env python
# coding: utf-8

from scipy.io import loadmat
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

mat_arr = loadmat('../data/imdb_crop/imdb.mat')['imdb']

face_coords = mat_arr['face_location'][0][0][0]

urls = mat_arr['full_path'][0][0][0]
urls = list(map(lambda url: '../data/imdb_crop/'+url[0], urls))

photo_taken = mat_arr['photo_taken'][0][0][0]

dob = mat_arr['dob'][0][0][0]

year_to_subtr = []
broken_idx = []

for i, matlab_datenum in enumerate(dob):
    try:
        dt = timedelta(days=int(matlab_datenum) -366) + datetime(1,1,1)
        dt_arr = list(dt.timetuple())
        year = dt_arr[0]

        # suppose that photo was taken at 1 of July
        if dt_arr[1] == 7:
            if dt_arr[2] > 1:
                year += 1
        if dt_arr[1] > 7:
            year += 1
        year_to_subtr.append(year)
    except:
        broken_idx.append(i)
    if (i % 10000) == 0:
         print("record", i+1, "processed, successfully parsed", len(year_to_subtr), 'years... ')

face_coords = np.delete(face_coords,broken_idx)
face_coords = list(map(lambda coords: coords[0], face_coords))

urls = np.delete(urls,broken_idx)

photo_taken = np.delete(photo_taken,broken_idx)

ages = list(map(lambda taken, yob: taken - yob, photo_taken, year_to_subtr))

def age_to_clusters(age):
    if age < 14:
        return 0
    if age < 26:
        return 1
    if age < 46:
        return 2
    if age < 66:
        return 3
    return 4

clusters = list(map(age_to_clusters, ages))

result = {}
result['urls'] = urls
result['face_coords'] = face_coords
result['age_cluster'] = clusters

df = pd.DataFrame(result, index=None)
print(df.shape)

with open('../csv/total.csv', mode='w', encoding='utf-8') as f_csv:
    df.to_csv(f_csv)
