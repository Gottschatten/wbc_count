# fetch dataset from keggle datasets download -d paultimothymooney/blood-cells

import os
import shutil
from zipfile import ZipFile
# remove directories if they exist ./train/, labels.csv, blood-cells.zip, dataset2-master and dataset-master
if os.path.exists('./train'):
    shutil.rmtree('./train')
if os.path.exists('./labels.csv'):
      os.remove('./labels.csv')
if os.path.exists('./blood-cells.zip'):
      os.remove('./blood-cells.zip')
if os.path.exists('./dataset2-master'):
      shutil.rmtree('./dataset2-master')
if os.path.exists('./dataset-master'):
      shutil.rmtree('./dataset-master')



# download dataset from kaggle
os.system('kaggle datasets download -d paultimothymooney/blood-cells')

# extract dataset
with ZipFile('blood-cells.zip', 'r') as zipObj:
   # Extract all the contents of zip file in current directory
   zipObj.extractall()



# mv direcory  ./dataset2-master/images/TRAIN to ./train
os.rename('./dataset2-master/dataset2-master/images/TRAIN', './train')
# mv ./dataset-master/lables.csv to ./labels.csv
os.rename('./dataset-master/dataset-master/labels.csv', './labels.csv')

# rename sub directories in ./train to lowercase
dirs = os.listdir('./train/')
for d in dirs:
    os.rename('./train/' + d, './train/' + d.lower())

# remove the zip file and datasets
os.remove('blood-cells.zip')
shutil.rmtree('./dataset-master')
shutil.rmtree('./dataset2-master')

   
print('Done!')

