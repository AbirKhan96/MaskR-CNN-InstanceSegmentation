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
"""
altitude
np. set_printoptions(suppress=True)"""
np. set_printoptions(suppress=True)
def get_angle_from_pxl_y(pxl_y, center_y, fov_y):
    fov_center = fov_y/2
    if (pxl_y<center_y):
        angle = fov_center*(1-(pxl_y/center_y))
    else:
        angle = -(fov_center*((pxl_y-center_y)/center_y))
    return angle

def get_point_altitude(source_alt, depth, angle_deg):
    return source_alt + (depth*math.sin((math.pi/180)*angle_deg))


"""
visualisation
"""
# f=  open ('Shape_file.csv', 'w')
# f.close()
def decimate(population, factor=160):
    """ jumps by factor """
    return population[::factor]

def plot(points, beg_point, end_point, collision_points, ray_points):
    """all single points are np.ndarray of shape (1, 3)"""

    # plotting for all points is expensive. 
    # sub-sample them
    sampled_points = decimate(points, factor=100)

    colors = np.array([(0,0,0, 0.03)]*len(sampled_points))

    beg_point_color = [0, 255, 0, 1]
    end_point_color = [0, 0, 255, 1]
    col_point_colors = [[255, 0, 0, 1]]*len(collision_points)
    ray_points_col = [[240,128,128, 0.4]]*(len(ray_points)-2)

    _new_ps = np.concatenate([sampled_points, [beg_point[0], end_point[0]], collision_points, ray_points[1:-1]])
    _new_cs = np.concatenate([colors, [beg_point_color, end_point_color], col_point_colors, ray_points_col])


    fig = go.Figure(data =[go.Scatter3d(
                                    x = _new_ps[:,0],
                                    y = _new_ps[:,1],
                                    z = _new_ps[:,2],
                                    mode ='markers',
                                    marker = dict(size = 2, color=_new_cs))])
    fig.show()

"""
equidistant points between two 3d points
"""

def lerp(v0, v1, i):
    return v0 + i * (v1 - v0)

def getEquidistantPoints(p1, p2, n):
    return np.array([(lerp(p1[0],p2[0],1./n*i), lerp(p1[1],p2[1],1./n*i), lerp(p1[2],p2[2],1./n*i)) for i in range(n+1)])



# from metadata (meters)
def get_target_xyz(depth, source_x, source_y, source_z, target_pixel_x, target_pixel_y, 
                   image_width, image_height, fov_x, fov_y, center_bearing):
    src_lon, src_lat = convert_XY_to_lat_lon(source_x, source_y, source_z)
    # print ("Bearing Input:",target_pixel_x, image_width, fov_x, center_bearing)
    target_bearing = CalculateBearingfromFOV(target_pixel_x, image_width, fov_x, center_bearing)
    # print ("Target Bearing:", target_bearing)
    target_lat, target_lon = CalculatelatlonfromBearing(src_lat, src_lon, target_bearing, depth)

    (target_x, target_y), target_z = (convert_lat_lon_to_XY(target_lon, target_lat, source_z),
                        get_point_altitude(source_alt=source_z, depth=depth, angle_deg=get_angle_from_pxl_y(target_pxl_y, image_height/2, fov_y)))

    return [target_x, target_y, target_z]


def inside_aoi(target_pxl_x, target_pxl_y, H,W):
    valid = True

    h_lim_top = int(0.05*H)
    h_lim_bot = H - int(0.20*H)

    w_lim_lft = int(0.20*W)
    w_lim_rgt = W - int(0.20*W)

    h_center = H//2
    w_center = W//2

    if target_pxl_x < w_lim_lft and target_pxl_x > w_lim_rgt:
        valid = False
        return valid
    if target_pxl_y < h_lim_top and target_pxl_y > h_lim_bot:
        valid = False
        return valid
        
    w_mid_lft = w_center - int(0.20*w_center)
    w_mid_rgt = w_center + int(0.20*w_center)

    if target_pxl_x > w_mid_lft and target_pxl_x < w_mid_rgt:
        valid  = False

    return valid

def first_colliding_point_idx_v1(kd_tree, beg_point: np.ndarray, end_point: np.ndarray, points, dist_between_points=0.05, min_collision_dist=0.5):
    """
    :@param beg_point: shape (1,3) - source location
    :@param end_point: shape (1,3) - point at maximum poissible distance from source
    :@param points: shape (1,n_points) - point cloud points
    """
    # n_equidistant_points = points_per_meter * total_num_of_meter
    n_equidistant_points = (1/dist_between_points) * int(np.linalg.norm(beg_point[0] - end_point[0]))
    #print('n_equidistant_points:', int(n_equidistant_points))
    equidistant_points = getEquidistantPoints(beg_point[0], end_point[0], int(n_equidistant_points))
    #print(equidistant_points)

    collision_idxs = []
    min_dist, min_idx = math.inf, -1
    for p in equidistant_points:
        dist, ind = kd_tree.query([p], k=1)
        if dist[0][0] < min_dist:
            min_dist = dist[0][0]
            min_idx = ind[0][0]
            # closer points to source will be @lower indices
            # and far away points to source will be @higher indices
            if min_dist<min_collision_dist:
                collision_idxs.append((min_idx, dist[0][0]))

    return collision_idxs, equidistant_points

ROOT_FOLDER = "/mnt/cg/01.Pegasus_Bulk Processed Data/Bulk/"
ROOT_FOLDER = "/mnt/PCMC/03.Output/Pegasus_Bulk Processed Data/"

MISSION_DIR_INPUT =  "2021-APR-09_pcmcMission1_53018"
MODEL_NAME = 'Manapa_Bhavan_Zone_15Aug'
ROOT_SAVE_FOLDER = "/home/itis/Desktop/PCMC_JPG/"
ROOT_SAVE_FOLDER = "/home/itis/Desktop/Work_Flow_PCMC/src/"
LOCATION = "Manapa_Bhavan_Zone"

# LIDAR_FOLDER = os.path.join(ROOT_FOLDER, MISSION_DIR,'FOR_Orbit','Las')
# LIDAR =  os.listdir(LIDAR_FOLDER)

start_time = time.time()

if True:#for MISSION_DIR_INPUT in os.listdir(os.path.join(ROOT_SAVE_FOLDER, MODEL_NAME)):
    if True:
        # with open(ROOT_SAVE_FOLDER+'/{}/{}_las_info.pickle'.format("Las_Info_MetaData",MISSION_DIR), 'rb') as handle:
        #     las_num_dict = pickle.load(handle)
        with open(ROOT_SAVE_FOLDER+'/{}_{}_metadata.pickle'.format(LOCATION,MISSION_DIR_INPUT), 'rb') as handle:
            metadata = pickle.load(handle)



        # image id is Image_name.split('.')[0]
        # metadata.keys() Image id of all images of all tracks
        # las_num_dict.keys() Image id of all images in all tracks

        # pixel_path_dir = os.path.join(ROOT_SAVE_FOLDER, MODEL_NAME, MISSION_DIR)

        # pixel_file = open (pixel_path_dir + 'pixel_info.csv', 'r')


        
        pixel_path_dir = os.path.join(ROOT_SAVE_FOLDER)
        pixel_file = open (pixel_path_dir + '/Pixel_info_{}.csv'.format(MODEL_NAME), 'r')




        # bin_dir, BIN_FOLDER, IMAGE_NAME, imageWidth, imageHeight, ASSET, point_str

        # /mnt/cg/01.Pegasus_Bulk Processed Data/Bulk/AI_output/km/2021-FEB-24_Mission1CGSML/Track_ASphere/eval_Track_ASpher


        # Track_A_20210225_044653 Profiler.zfs_0 Sample LAs File Name
        Track_dict = dict()
        for line in pixel_file:
            # ROOT_SAVE_FOLDER, MODEL_NAME, MISSION_DIR,IMAGE_FOLDER, BIN_FOLDER, IMAGE_NAME, IMAGE_WIDTH, IMAGE_HEIGHT, ASSET, point_str = line.split('\t')
            bin_dir, BIN_FOLDER, MISSION_FOLDER,IMAGE_NAME, imageWidth, imageHeight, ASSET, point_str = line.split('\t')
            IMAGE_FOLDER = IMAGE_NAME.split('-')[0]#bin_dir.split('/')[-2]

            if IMAGE_FOLDER not in Track_dict:
                Track_dict[IMAGE_FOLDER] = []
            else:
                Track_dict[IMAGE_FOLDER].append(line)

        pixel_file.close()


        print (Track_dict.keys())
        # dfdf
        # print (las_num_dict.keys())
        curr_time = time.time()
        all_hits = []
        image_hit = []
        car_loc = []
        # Track_A_20210225_044653 Profiler.zfs_0 Sample LAs File Name

        line_count = 0
        # For_Orbit/Manapa_Bhavan_Zone/2021-APR-09_pcmcMission1_53018
        for track_key in Track_dict:
            try:
                print (track_key)
                # if 'TrackZ_C' not in track_key:
                #     continue
                lines = Track_dict[track_key]
                las_loaded = {}
                track_id  =track_key.replace('Sphere', '')
                point_cloud = []
                points_stack = []

                for root, dirs, files in os.walk(os.path.join(ROOT_FOLDER, 'For_Orbit',LOCATION, MISSION_DIR_INPUT, 'Las')):
                    for file in files:
                        if file.endswith(".las") and track_id in file:
                            las = os.path.join(root, file)
                            print (las)

                # for imageid_in_las in las_num_dict[track_id]:
                #     for las in las_num_dict[track_id][imageid_in_las]:
                #         if not las in las_loaded:
                #             print (las)
                            LAS_FILE_PATH =  las #os.path.join(LIDAR_FOLDER, las)
                            point_cloud1=lp.file.File(LAS_FILE_PATH, mode="r")
                            point_cloud.append(point_cloud1)
                            points1 = np.vstack((point_cloud1.x, point_cloud1.y, point_cloud1.z)).transpose()
                            print (points1.shape)
                            points_stack.append(points1)
                            las_loaded[las] = 1

                print ("here")
                points = np.concatenate(points_stack)
                print (points.shape)
                kdtree = KDTree(points)

                print ("KD Tree Completed")
            except:
                continue
            line_num = 0
            for line in tqdm(lines):
                # print (line)
                try:
                    # ROOT_SAVE_FOLDER, MODEL_NAME, MISSION_DIR,IMAGE_FOLDER, BIN_FOLDER, IMAGE_NAME, IMAGE_WIDTH, IMAGE_HEIGHT, ASSET, point_str = line.split('\t')
                    bin_dir, BIN_FOLDER, MISSION_FOLDER,IMAGE_NAME,  IMAGE_WIDTH, IMAGE_HEIGHT, ASSET, point_str = line.split('\t')
                    MISSION_DIR, IMAGE_FOLDER, BIN_FOLDER = bin_dir.split('/')[-3],bin_dir.split('/')[-2], bin_dir.split('/')[-1] 
                    # if MISSION_FOLDER != MISSION_DIR_INPUT:
                    #     continue
                    print (MISSION_FOLDER, IMAGE_FOLDER, track_key)


                    point_str = point_str[:-1]
                    if point_str == "":
                        continue
                    image_id = IMAGE_NAME#.split('.')[0]
                    #print (line)
                    track_id = image_id.split('-')[0]
                    # print (las_num_dict[track_id][image_id], track_id, image_id)
                    # for las in las_num_dict[track_id][image_id]:
                    #     LAS_FILE_PATH =  os.path.join(LIDAR_FOLDER, las)
                    #     point_cloud1=lp.file.File(LAS_FILE_PATH, mode="r")
                    #     point_cloud.append(point_cloud1)
                    #     points1 = np.vstack((point_cloud1.x, point_cloud1.y, point_cloud1.z)).transpose()
                    #     print (points1.shape)
                    #     points_stack.append(points1)

                    # print ("here")
                    # points = np.concatenate(points_stack)
                    # print (points.shape)
                    # kdtree = KDTree(points)
                    # print ("here 2")
                    # COLLISION POINTS ACCUMULATOR
                    collision_points = dict()
                    image_id = IMAGE_NAME#.split('.')[0]
                    # print (image_id)
                    # image metadata
                    image_width = int(IMAGE_WIDTH) # x pixel max
                    image_height = int(IMAGE_HEIGHT) # y pixel max
                    # 360 -20.90897959 -180
                    fov_x = 360 # around x pixel (degrees)
                    fov_y = 180 # around y pixel (degrees)

                    # lidar source (meters)
                    source_x = float(metadata[image_id][0])
                    source_y = float(metadata[image_id][1])
                    source_z = float(metadata[image_id][2])
                    # center_bearing = 360 - float(metadata[image_id][3]) -180
                    center_bearing = float(metadata[image_id][3])
                    present_car_loc = [source_x, source_y, source_z]
                    car_loc.append(present_car_loc)

                    # print (source_x, source_y, source_z, center_bearing)

                    hits = []
                    random_mask_pxls = []
                    point_str = point_str[:-1]
                    for point in point_str.split(';')[0:1]:

                        target_pxl_x = int(point.split(',')[0])
                        target_pxl_y = int(point.split(',')[1])

                        # if not inside_aoi(target_pxl_x, target_pxl_y, image_height, image_width):
                        #     print ("Not Inside AOI")
                        #     continue
                        # random_mask_pxls.append((target_pxl_x, target_pxl_y))

                        depth =  15

                        dist_between_points = 0.05 # for points made on line from lidar source to point at `depth`
                        min_collision_dist = 0.25 # min required distance between a point cloud and equidistant points on line

                        beg_point = [source_x, source_y, source_z] # from metadata (meters)
                        # print (IMAGE_NAME, ASSET)
                        end_point = get_target_xyz(depth, source_x, source_y, source_z, target_pxl_x, target_pxl_y, 
                                        image_width, image_height, fov_x, fov_y, center_bearing) # heavily relies on metadata (meters)
                        # print (end_point)



                        collision_idxs, equidistant_points = first_colliding_point_idx_v1(kd_tree=kdtree, 
                                            beg_point=np.array([beg_point]),
                                            end_point=np.array([end_point]),
                                            points=points,
                                            dist_between_points=dist_between_points,
                                            min_collision_dist=min_collision_dist)

                        if len(collision_idxs) > 0:
                            hits.append(points[collision_idxs[0][0]])


                        else:
                            print('missed')
                        # print (equidistant_points)
                        # with open("equidistant_points_{}.csv".format(track_key,), "ab") as ef:
                        #     np.savetxt(ef, equidistant_points, delimiter = ",", fmt='%f')
                    # hits =np.median(hits, axis=0)
                    all_hits.append(hits)
                    # np.savetxt("Shape_file.csv", hits, delimiter=',')
                    # print (hits.shape, line_num)
                    line_num = line_num + 1
                    # try:
                    #     with open("Entry_{}.csv".format(track_key), "a") as ef:
                    #         np.savetxt(ef, [hits],delimiter = ",", fmt='%f')
                    # except:
                    #     pass


                    # if len (hits) >0:
                    #     to_write = '{},{},'.format(ASSET,IMAGE_NAME)
                    #     for value in hits[0]:
                    #         to_write = to_write + str(value)+','
                    #     to_write = to_write[:-1]

                    #     print(hits, "hit ho gaya congratulations of car location ", present_car_loc)            
                    #     shp =  open("Shape_file_{}_{}.csv".format(track_key,curr_time), "a")
                    #     shp.write(to_write+"\n")
                    #     shp.close()

                    if len (hits) > 0:
                        with open("{}_Equidistant_Points_{}_{}.csv".format(MODEL_NAME, curr_time ,MISSION_DIR_INPUT), "a") as ef:
                            np.savetxt(ef, equidistant_points, delimiter = ",", fmt='%f')
                        to_write = '{},{},{},'.format(ASSET,IMAGE_NAME,MISSION_FOLDER)
                        for value in hits[0]:
                            to_write = to_write + str(value)+','
                        to_write = to_write[:-1]
                        # print(hits, "hit ho gaya congratulations of car location ", present_car_loc)            
                        shp =  open("{}_ShapeFile_{}_{}.csv".format(MODEL_NAME,curr_time, MISSION_DIR_INPUT), "a")
                        shp.write(to_write+"\n")
                        shp.close()

                        to_write = '{},{},{},'.format(ASSET,IMAGE_NAME,MISSION_FOLDER)
                        line_obj = LineString([Point(beg_point[0], beg_point[1], beg_point[2]), Point(end_point[0], end_point[1], end_point[2])]).wkt
                        # for value in beg_point:
                        #     to_write =  to_write + str(value)+ ','

                        # for value in end_point:
                        #     to_write =  to_write + str(value)+ ','
                        # to_write = to_write[:-1]
                        shp =  open("{}_Line_ShapeFile_{}_{}.csv".format(MODEL_NAME,curr_time, MISSION_DIR_INPUT), "a")
                        shp.write(line_obj+"\n")
                        shp.close()
                            

                        line_count = line_count +1
                        if line_count ==100000000:
                            break



                except Exception as E:
                    print (E)
                    traceback.print_exc()
                    pass



end_time = time.time()

print ("Time Taken : ", end_time - start_time)
print ("Completed Lidar Process .")
# # with open("equidistant_points.csv", "ab") as f:
# #     np.savetxt(f, equidistant_points, delimiter = ",", fmt='%f')
# np.savetxt("Shape_file.csv", all_hits, delimiter = ",", fmt="%f")

# np.savetxt("Car_loc_Shape_file.csv", car_loc, delimiter = ",", fmt="%f")

# dsa



