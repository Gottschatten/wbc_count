# fetch dataset from keggle datasets download -d paultimothymooney/blood-cells

import os
from zipfile import ZipFile


# download dataset from kaggle
os.system('kaggle datasets download -d paultimothymooney/blood-cells')

# extract dataset
with ZipFile('blood-cells.zip', 'r') as zipObj:
   # Extract all the contents of zip file in current directory
   zipObj.extractall()
   
print('Done!')

