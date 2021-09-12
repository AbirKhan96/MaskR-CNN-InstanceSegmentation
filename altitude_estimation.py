import time
import os
import traceback
import pickle
import numpy as np
import math
import utm # for generating _lat_lon 
import pyproj
import plotly.express as px
import plotly.graph_objects as go
import laspy as lp
import json
from tqdm import tqdm
from pprint import pprint
from sklearn.neighbors import KDTree
from shapely.geometry import Point
from shapely.ops import transform
from utils import *
from shapely.geometry import Point, LineString
import pandas
"""
altitude
np. set_printoptions(suppress=True)"""
np. set_printoptions(suppress=True)


ROOT_FOLDER = "/mnt/cg/01.Pegasus_Bulk Processed Data/Bulk/"
ROOT_FOLDER = "/mnt/PCMC/03.Output/Pegasus_Bulk Processed Data/"

MISSION_DIR_INPUT =  "2021-APR-09_pcmcMission1_53018"
MODEL_NAME = 'Manapa_Bhavan_Zone_16Aug'
ROOT_SAVE_FOLDER = "/home/itis/Desktop/PCMC_JPG/"

LOCATION = "Manapa_Bhavan_Zone"


# LIDAR_FOLDER = os.path.join(ROOT_FOLDER, MISSION_DIR,'FOR_Orbit','Las')
# LIDAR =  os.listdir(LIDAR_FOLDER)


if True:#for MISSION_DIR_INPUT in os.listdir(os.path.join(ROOT_SAVE_FOLDER, MODEL_NAME)):
    if True:
        # with open(ROOT_SAVE_FOLDER+'/{}/{}_las_info.pickle'.format("Las_Info_MetaData",MISSION_DIR), 'rb') as handle:
        #     las_num_dict = pickle.load(handle)
        with open(ROOT_SAVE_FOLDER+'/{}_{}_metadata.pickle'.format(LOCATION,MISSION_DIR_INPUT), 'rb') as handle:
            metadata = pickle.load(handle)



vector_file = "Manapa_Bhavan_Zone_16Aug_ShapeFile_1629184655.3826542_2021-APR-09_pcmcMission1_53018.csv"

new_vector_file = "Manapa_Bhavan_Zone_16Aug_ShapeFile_1629184655.3826542_2021-APR-09_pcmcMission1_53018_altitude.csv"



fread = open(vector_file, 'r')
fwrite = open(new_vector_file, 'w')

for line in fread:
    data = line.split(',')
    new_line = line.strip() +',' +str(float(data[5]) - metadata[data[1]][2]) +'\n'
    fwrite.write(new_line)
    

fread.close()
fwrite.close()