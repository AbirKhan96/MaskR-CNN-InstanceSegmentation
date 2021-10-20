import os, json, io, base64, cv2, glob
from pprint import pprint
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import labelme
import os
import joblib
import numpy
import copy
from loguru import logger

from tqdm import tqdm
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode


from pipeline.train.det2.trainer import Det2Trainer
from pipeline.eval.det2.get_model import GetTrained

# trainer = Det2Trainer(
#   data=DataConfig.AllowedClassesDataset,
#   model=ModelConfig.Allowed_ClassesModel,
#   cfg=TrainConfig)

from skimage.morphology import skeletonize
from skimage import data
import matplotlib.pyplot as plt
from skimage.util import invert

def dump(obj, saveby='./temp.bin'):
	ret = joblib.dump(obj, saveby)
	logger.debug(f"saved {ret}")

def load(from_path):
	return joblib.load(from_path)
def load_bin(from_path):
	return joblib.load(from_path)
import numpy as np
kernel = np.ones((5,5), np.uint8)
def get_seg_dict(predictor, on_im, save_dir, fname, THING_CLASSES):

	im = cv2.imread(on_im)
	# format at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
	outputs = predictor(im)

	dic = outputs['instances'].__dict__

	pred_class_list =list(dic['_fields']['pred_classes'].cpu().numpy())
	del dic['_fields']['pred_classes']
	dic['_fields']['pred_classes'] = []

	print(len(pred_class_list), pred_class_list)
	print("#"*80)
	for obj in pred_class_list:
		dic['_fields']['pred_classes'].append(THING_CLASSES[obj])

	dic['_fields']['pred_classes'] = numpy.array(dic['_fields']['pred_classes'])
	v = Visualizer(im[:, :, ::-1],
			metadata=metadata, 
			scale= 1.0, 
			instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
	)
	out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
	#plt.imshow(out.get_image()[:, :, ::-1])

	## To save image please un comment below line.
	dic['_fields']['pred_classes']
	img = out.get_image()[:, :, ::-1]
	# img = cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_RGB2BGR)
	cv2.imwrite(save_dir+'/'+fname, img)

	mask = dic['_fields']['pred_masks'].cpu().detach().numpy()
	print (mask.shape)

	points_list = []
	for objects_num in range(0,mask.shape[0]):
		try:
			new_mask = mask[objects_num]
			print (new_mask.shape)
			# new_mask = cv2.erode(np.float32(new_mask), kernel, iterations=0)
			# new_mask = invert(new_mask)
			# new_mask = skeletonize(new_mask)
			print ("here")
			new_mask = new_mask.astype(np.uint8)  #convert to an unsigned byte
			new_mask*=255
			print (np.unique(new_mask))
			
			cnts = cv2.findContours(new_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			cnts = cnts[0] if len(cnts) == 2 else cnts[1]
			c = max(cnts, key=cv2.contourArea)
			left = tuple(c[c[:, :, 0].argmin()][0])
			right = tuple(c[c[:, :, 0].argmax()][0])
			top = tuple(c[c[:, :, 1].argmin()][0])
			bottom = tuple(c[c[:, :, 1].argmax()][0])
			# points_indices = points.shape[0]
			# random_indices = np.random.choice(points_indices, size=1)
			# points = points[random_indices]

			points = [bottom]
			print ("NewPoints")

			print (points)
			obj_name = dic['_fields']['pred_classes'][objects_num]

			points_list.append((obj_name,points))
		except Exception as E:
			print (E)
			pass



	# try:
	# 	points = np.argwhere(mask==True)
	# 	points_indices = points.shape[0]
	# 	random_indices = np.random.choice(points_indices, size=5)
	# 	points = points[random_indices]
	# 	points = np.expand_dims(points, axis=(1))
	# except:
	# 	points = np.array([])
	# print ("%%"*80)
	# print (points, fname)
	
	dic = {
		'imgShape': dic['_image_size'],
		'predClasses': dic['_fields']['pred_classes'],
		'predBoxes': dic['_fields']['pred_boxes'].tensor.cpu().detach().numpy(),
		'boxScores': dic['_fields']['scores'].cpu().detach().numpy(),
		'instance_predictions': points_list, #dic['_fields']['pred_masks'].cpu().detach().numpy(),
	}
	return dic



def get_im_seg_info(images_dir, predictor, save_dir, THING_CLASSES, ext='jpg'):
	
	#  vis_metadata = load_bin(from_path=vis_metadata)  # train metadata

	write_dir = save_dir+'/'+ f"eval_{images_dir[:-1].split('/')[-1]}" + '/'
	logger.debug(f"saving segmented images in {write_dir}")
	os.makedirs(write_dir, exist_ok=True)

	# check..
	# logger.debug(vis_metadata.thing_classes)

	#list_of_dics = []
	import time
	start_time = time.time()
	print (start_time)
	for im_name in tqdm(os.listdir(images_dir)[150:600]):
		print (im_name)
		if os.path.exists(write_dir + f'{im_name}._info.bin'):
			continue
		if im_name.split('.')[-1].lower() == ext.lower():
			
			dic = get_seg_dict(
				predictor=predictor,
				on_im=os.path.join(images_dir,im_name),
				save_dir = save_dir,
				fname= im_name,
				THING_CLASSES=THING_CLASSES
			)
			
			#from pprint import pprint
			#pprint(dic_str)
			dic['imageName'] = im_name
			joblib.dump((copy.deepcopy(dic), THING_CLASSES), write_dir + f'{im_name}._info.bin')
			#list_of_dics.append(dic)
	end_time = time.time()


	print ("Total Time Taken : ",end_time- start_time)
			

	return None   



from config import TrainConfig, DataConfig, ModelConfig, DataPreparationConfig, MODEL_NAME, MODEL_BASE

# from config_elect_pole import TrainConfig, DataConfig, ModelConfig, DataPreparationConfig, MODEL_NAME, MODEL_BASE

print (MODEL_NAME)


THING_CLASSES = DataConfig.AllowedClassesDataset.thing_classes
metadata = MetadataCatalog.get("dataset_train")

print(THING_CLASSES)

gpu_device = 'cuda:0'
NUMBER_OF_CORE = 1
predictor, cfg = (
		GetTrained(ModelConfig.Allowed_ClassesModel.__name__.replace("Model", ""), gpu_device, base_dir = str(MODEL_BASE))
		.fetch(thresh=0.2, cfg=True))

# get_im_seg_info("/mnt/PCMC/03.Output/Pegasus_Bulk Processed Data/For_Orbit/Manapa_Bhavan_Zone/2021-APR-09_pcmcMission1_53018/Images/JPG/", predictor, "Manapa_Bhavan_Zone_26_9_NEW", THING_CLASSES, ext='jpg')


# FOR GPU 0

SECTOR5_FOLDERS = [
				
                                "20210622/1831/01657-1624335208"		
 					]

# FOR GPU 1

# SECTOR5_FOLDERS = [
# 						"20210601/1831/01657-1622528438",
# 						"20210601/1831/01657-1622530398"
# 					]

# get_im_seg_info("/mnt/thane/03.Output/06. Pano/02. I-Star/20210601/1831/01657-1622542644", predictor, "20210601/1831/01657-1622542644", THING_CLASSES, ext='jpg')

def process_img_dir(foldername):
	get_im_seg_info("/mnt/thane/03.Output/06. Pano/02. I-Star/"+foldername, predictor, "OutputSector5/"+foldername, THING_CLASSES, ext='jpg')

# from multiprocessing import Pool
from torch.multiprocessing import Pool, Process, set_start_method


try:
	set_start_method('spawn')
	with Pool(NUMBER_OF_CORE) as p:
		print(p.map(process_img_dir, SECTOR5_FOLDERS))
except RuntimeError:
	pass

# ROOT_FOLDER = "/mnt/cg/01.Pegasus_Bulk Processed Data/Bulk/"
# ROOT_SAVE_FOLDER = "/mnt/cg/01.Pegasus_Bulk Processed Data/Bulk/AI_output/"

#ROOT_FOLDER = "/home/itis/jaipur new las/PANO/Track_A.Ladybug/"
#ROOT_SAVE_FOLDER = "/home/itis/jaipur new las/PANO/AI_OUTPUT/"

#ROOT_SAVE_FOLDER = os.path.join(ROOT_SAVE_FOLDER, MODEL_NAME)



# TRACK_FOLDER = ""
# IMAGE_FOLDER = ""
# SAVE_FOLDER = ""

# for MISSION_FOLDER in tqdm(os.listdir(ROOT_FOLDER)):
# 	if 'Mission' in MISSION_FOLDER:
# 		Mission =  MISSION_FOLDER
# 		if '2021-FEB-24_Mission1CGSML' in MISSION_FOLDER:
# 			continue
# 		#print (MISSION_FOLDER)
# 		MISSION_FOLDER = os.path.join(ROOT_FOLDER, MISSION_FOLDER, "For_Orbit")
# 		for TRACK_FOLDER in tqdm(os.listdir(MISSION_FOLDER)):
			
# 			if "Track" in TRACK_FOLDER and 'Sphere' in TRACK_FOLDER:
# 				#print ("\t------->", TRACK_FOLDER)
# 				IMAGE_FOLDER = os.path.join(ROOT_FOLDER, MISSION_FOLDER, TRACK_FOLDER) 
# 				SAVE_FOLDER =  os.path.join(ROOT_SAVE_FOLDER, Mission, TRACK_FOLDER)
# 				print (SAVE_FOLDER)
# 				if not os.path.exists(SAVE_FOLDER):
# 					os.makedirs(SAVE_FOLDER)
# 				for images in os.listdir(os.path.join(ROOT_FOLDER, MISSION_FOLDER, TRACK_FOLDER)):
# 					print ("\t\t --------->", images)
# 				get_im_seg_info(IMAGE_FOLDER, predictor, SAVE_FOLDER, THING_CLASSES, ext='jpg')
				
# 				break

# 		break
