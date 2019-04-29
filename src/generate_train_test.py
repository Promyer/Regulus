#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

rel_path = '../' ##
data = pd.read_csv(rel_path+'csv/total.csv', index_col=0)
print(data.shape)

import os
croped_fnames = os.listdir(rel_path+'data/croped')
print('Total croped photos:', len(croped_fnames))

old_fnames = []
young_fnames = []

croped = data.loc[map(lambda url: ('crop_'+url.split('/')[-1]) in croped_fnames, data['urls'])]

print
print('Classes distribution in croped photos:')
print('   ', np.unique(croped['age_clusters'], return_counts=True))

croped_young_old = croped.loc[croped['age_clusters'] != 2]

y = croped_young_old.age_clusters
X = croped_young_old.drop('age_clusters', axis=1)

# train-test 9:1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
print
print('Train size:', len(X_train))
print('  classes distribution:', np.unique(y_train, return_counts=True))
print('Test size:', len(X_test))
print('  classes distribution:', np.unique(y_test, return_counts=True))

src_folder = '../../../../data/croped/'
dst_folder = rel_path + 'datasets/young2old/train/'

for i, row in enumerate(X_train.itertuples()):
    photo_url = 'crop_' + row.urls.split('/')[-1]
    if y_train.iloc[i] == 0:
        dst_class_folder = 'A/'
    else:
        dst_class_folder = 'B/'
    os.symlink(src_folder + photo_url, dst_folder + dst_class_folder + photo_url)
print('Train folders generated')


dst_folder = rel_path + 'datasets/young2old/test/'

for i, row in enumerate(X_test.itertuples()):
    photo_url = 'crop_' + row.urls.split('/')[-1]
    if y_test.iloc[i] == 0:
        dst_class_folder = 'A/'
    else:
        dst_class_folder = 'B/'
    os.symlink(src_folder + photo_url, dst_folder + dst_class_folder + photo_url)   
print('Test folders generated')