import sys
sys.path.append("/home/claudio/Documents/GitHub/datascienceworkshop-pneumonia")


from config import TEST_DCM, TRAIN_DCM, TRAIN_IMAGES, TEST_IMAGES, DATA_DIR
import pydicom
from PIL import Image
import pandas as pd
import os
import shutil


metadata = pd.read_csv(os.path.join(TRAIN_IMAGES,r'metadata.csv'))
print(metadata.head())
grouped = metadata.groupby(by='patient_id')
print('\ngroups are: ' + str(grouped.ngroups))

positive_files = os.listdir(os.path.join(TRAIN_IMAGES,'positive'))
negative_files = os.listdir(os.path.join(TRAIN_IMAGES,'negative'))
print('positive files are: ' + str(len(positive_files)))
print('negative files are: ' + str(len(negative_files)))
fn=positive_files[:]
fn.extend(negative_files)
filenames = [s.split('.')[0] for s in fn]
print('\n')
print('filenames are: ' + str(len(filenames)))
print(filenames[:5])
print('\n')

def InList(x):
    return x in filenames

pId_hashmap = pd.DataFrame()
pId_hashmap['hash'] = metadata['patient_id'].apply(func=InList) 

print(pId_hashmap.head())
response = pId_hashmap.eq(True).all()
print('True if all values are True: ' + str(response))
