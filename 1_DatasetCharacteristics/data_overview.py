# load data from 'labels.csv'
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os


# clean up directory
if os.path.exists('distribution.png'):
    os.remove('distribution.png')
if os.path.exists('distribution_num.png'):
    os.remove('distribution_num.png')


# Load data
data = pd.read_csv('labels.csv')

# remove unnamed column
data = data.drop('Unnamed: 0', axis=1)

# display the shape of the data
print(data.shape)

# display the first few rows of the data and the structure of the data

# plot the distribution of 'Image' for 'Category'
plt.figure(figsize=(12, 8))
sns.countplot(data['Category'])
plt.title('Distribution of Categories')
# save the plot
plt.savefig('distribution.png')

# get the list of directories in ./train/
cells = {'neutrophil': 0, 'eosinophil': 0, 'lymphocyte': 0, 'monocyte': 0}
dirs = os.listdir('./train/')
for d in dirs:
    if d in cells.keys():
        cells[d] = len(os.listdir('./train/' + d))

# plot the distribution of the number of images in each category
sns.barplot(x=list(cells.keys()), y=list(cells.values()))
plt.ylim(2300, 2500)
plt.ylabel('Number of Images')
plt.xlabel('Category')
plt.yticks(np.arange(2250, 2500, 25))
plt.title('Distribution of the Number of Images in Each Category')
# save the plot
plt.savefig('distribution_num.png')



