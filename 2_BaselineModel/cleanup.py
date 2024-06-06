# sort images in ../sourcedata/dataset-master/dataset-master/JPEGImages/ into five folders in ../sourcedata/images_simple/ according to their labels in ../sourcedata/dataset-master/dataset-master/labels.csv labeld 'Image' and 'Category

import os
import pandas as pd
import shutil



# read the csv file
df = pd.read_csv('../sourcedata/dataset-master/dataset-master/labels.csv')
# get the unique categories
categories = df['Category'].unique()

# 'Image' label transform to 5-digit number

for index, row in df.iterrows():
    image = row['Image']
    if image =< 9:
        image = '0000' + image
        continue    
    if image =< 99:
        image = '000' + image
        continue
    if image =< 999:
        image = '00' + image
        continue
    if image =< 9999:
        image = '0' + image
        continue
df.at[index, 'Image'] = image


# create the folders
for category in categories:
    os.makedirs('../sourcedata/images_simple/' + category, exist_ok=True)

# move the images to the folders
for index, row in df.iterrows():
    image = row['Image']
    category = row['Category']
    shutil.copy('../sourcedata/dataset-master/dataset-master/JPEGImages/' + image, '../sourcedata/images_simple/' + category + '/' + image)
    print('Moved ' + image + ' to ' + category)

print('Done!')
