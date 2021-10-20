import os
print ("ello")



# get_ipython().system('nvidia-smi')


# Some basic setup:
# Setup detectron2 logger
import torch
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import os, json, cv2, random
from tqdm import tqdm
from pprint import pprint
from pathlib import Path
import numpy as np
from PIL import Image
import joblib
import matplotlib.pyplot as plt
# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode

from detectron2.engine import DefaultTrainer
from detectron2.engine import HookBase
from detectron2.data import build_detection_train_loader
import detectron2.utils.comm as comm

debug=False


import wandb
wandb.login()
wandb.init(project='PCMC_TEST', sync_tensorboard=True)

# import wandb

# # 1. Start a new run
# wandb.init(project='gpt-3', entity='abir96')

# from detectron2.data.datasets import register_coco_instances

# register_coco_instances("dataset_train", {}, "/home/itis/Desktop/Work_Flow_PCMC_rakesh/src/store/data/Allowed_Classes/train.json", "/home/itis/Desktop/Work_Flow_PCMC_rakesh/src/store/data/Allowed_Classes/train")
# register_coco_instances("dataset_val", {}, "/home/itis/Desktop/Work_Flow_PCMC_rakesh/src/store/data/Allowed_Classes/test.json", "/home/itis/Desktop/Work_Flow_PCMC_rakesh/src/store/data/Allowed_Classes/test")
# register_coco_instances("dataset_test", {}, "/home/itis/Desktop/Work_Flow_PCMC_rakesh/src/store/data/Allowed_Classes/holdout.json", "/home/itis/Desktop/Work_Flow_PCMC_rakesh/src/store/data/Allowed_Classes/holdout")
from pipeline.train.det2.trainer import Det2Trainer
from config import TrainConfig, DataConfig, ModelConfig

# configure trainer
trainer = Det2Trainer(
  data=DataConfig.AllowedClassesDataset,
  model=ModelConfig.Allowed_ClassesModel,
  cfg=TrainConfig)

class DataPreparationConfig:
    # while training all are true
    proc_json_files = True
    rm_dups = True # todo: checks
    train_test_split = True
    labelme2coco = True # needed for kpis only if files train, test, holdout contensts change
    reg_datasets = True # needed for kpis


# process the json files into trainable data
trainer.prepare_data(DataPreparationConfig)

# In[4]:


# REGISTER DATASET


# In[3]:


# TOD0: UNIQUE LABELS AND CLASS DISTRIBUITION
# distribution = {}
# for dirname in ["test", "holdout", "train"]:
#     labels = []
#     json_paths = [*(Path("store/data/Allowed_Classes/")/dirname).glob("*.json")]
#     for path in tqdm(json_paths, total=len(json_paths)):
#         with open(path, "r") as f:
#             ann = json.loads(f.read())
#             for shape in ann['shapes']:
#                 labels.append(shape["label"])
#     vals, cnts = np.unique(labels, return_counts=True)
#     distribution[dirname] = dict(labels=vals, counts=cnts)
    
#     fig=plt.figure(figsize=(20, 5))
#     plt.bar(vals, cnts)
#     plt.title(f"{dirname} unique: {len(vals)}")
#     plt.xticks(rotation=90, ha='right')
#     plt.grid()
#     plt.show()
#     plt.close()
    
unique_labels = [
    'hoardings', 
    'garbage_bins', 
    'transformer', 
    'bus_stops', 
    'fire_hydrants', 
    'traffic_sign', 
    'traffic_signal', 
    'strom_water_vent', 
    'electrical_pole', 
    'manhole_circular', 
    'feeder_pillar', 
    'manhole_rectangular',
    'footpath_gate', 
    'street_light', 
    'trees']


# In[17]:


def get_data_dicts(img_dir):
    img_dir = Path(img_dir)
    dataset_dicts = []
    jpg_paths = [*img_dir.glob("*.JPG")]
    for idx, im_path in tqdm(enumerate(jpg_paths), total=len(jpg_paths)):
        ann_path = str(im_path).replace(".JPG", ".json")
        with open(ann_path, "r") as f:
            ann = json.loads(f.read())
            # print(ann.keys())
            # ['version', 'flags', 'shapes', 'imagePath', 'imageData', 'imageHeight', 'imageWidth', 'imageHeightRatio', 'imageWidthRatio']
            del ann["imageData"] # save some ram
        
        record = {}
        # filename = os.path.join(img_dir, v["filename"])
        # height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = str(im_path) 
        record["image_id"] = idx
        record["height"] = ann["imageHeight"]
        record["width"] = ann["imageWidth"]
        
        objs = []
        for shape in ann["shapes"]:
            if shape['shape_type'] != 'polygon':
                continue
            poly = np.array(shape['points'])
            label = shape['label']
            
            cat_id = None
            for idx, ul in enumerate(unique_labels):
                if ul==label:
                    cat_id=idx
                    break
            if cat_id is None:
                    print("warning! unknown label present in data")
                    
            obj = {
                "bbox": [np.min(poly[:, 0]), np.min(poly[:, 1]), np.max(poly[:, 0]), np.max(poly[:, 1])],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [ [p for x in poly for p in x] ],
                "category_id": cat_id,
            }
            objs.append(obj)
      
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

def register_datasets(split_dirs: list, base_dir: str = "store/data/Allowed_Classes/"):
    for d in split_dirs:
        DatasetCatalog.register("pcmc_" + d, lambda d=d: get_data_dicts(base_dir + d))
        MetadataCatalog.get("pcmc_" + d).set(thing_classes=unique_labels)


# In[8]:


register_datasets(["train", "test", "holdout"])


# In[9]:


def visualize_registered_data(registered_name, dir_path, num_images=50, display_images=False, write_dir=None):
    metadata = MetadataCatalog.get(registered_name)
    dataset_dicts = get_data_dicts(dir_path)
    sample_size=min(len(dataset_dicts), num_images)
    Path(write_dir).mkdir(exist_ok=True, parents=True)
    for d in tqdm(random.sample(dataset_dicts, sample_size), total=sample_size):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
        out = visualizer.draw_dataset_dict(d)

        img = Image.fromarray(np.uint8(out.get_image()[:, :, ::-1])).convert('RGB')
        if display_images:
            display(img)
            
        if write_dir:
            img.save(str(Path(write_dir)/d["file_name"].split("/")[-1]))


# In[10]:


if debug:
    visualize_registered_data(registered_name="pcmc_test", dir_path="store/data/Allowed_Classes/test",
                              num_images=5, display_images=True, write_dir='temp/test')
    
    visualize_registered_data(registered_name="pcmc_holdout", dir_path="store/data/Allowed_Classes/holdout",
                          num_images=20, display_images=False, write_dir='temp/holdout')
    
    visualize_registered_data(registered_name="pcmc_train", dir_path="store/data/Allowed_Classes/train",
                          num_images=20, display_images=False, write_dir='temp/train')




# CLASS DISTRIBUITION
if debug:
    distribution = {}
    for dirname in ["test", "holdout", "train"]:
        labels = []
        json_paths = [*(Path("store/data/Allowed_Classes/")/dirname).glob("*.json")]
        for path in tqdm(json_paths, total=len(json_paths)):
            with open(path, "r") as f:
                ann = json.loads(f.read())
                for shape in ann['shapes']:
                    labels.append(shape["label"])
        vals, cnts = np.unique(labels, return_counts=True)
        distribution[dirname] = dict(labels=vals, counts=cnts)

        fig=plt.figure(figsize=(20, 5))
        plt.bar(vals, cnts)
        plt.title(f"{dirname} unique: {len(vals)}")
        plt.xticks(rotation=90, ha='right')
        plt.grid()
        plt.show()
        plt.close()


# In[12]:


class ValidationLoss(HookBase):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.clone()
        self.cfg.DATASETS.TRAIN = cfg.DATASETS.VAL
        self._loader = iter(build_detection_train_loader(self.cfg))
        
    def after_step(self):
        data = next(self._loader)
        with torch.no_grad():
            loss_dict = self.trainer.model(data)
            
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {"val_" + k: v.item() for k, v in 
                                 comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                self.trainer.storage.put_scalars(total_val_loss=losses_reduced, 
                                                 **loss_dict_reduced)


# In[ ]:


gpu_id="0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
OUTPUT_DIR = "./output_gpu"+gpu_id+"_TTP_PCMC"
MODEL_CONFIG = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(MODEL_CONFIG))
cfg.DATASETS.TRAIN = ("pcmc_train",)
cfg.DATASETS.VAL = ("pcmc_test",)
cfg.DATASETS.TEST = ("pcmc_holdout",)
cfg.DATALOADER.NUM_WORKERS = 32
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL_CONFIG)
cfg.SOLVER.IMS_PER_BATCH = 5  #16    #5
cfg.SOLVER.BASE_LR = 0.001  
cfg.SOLVER.MAX_ITER = 120000     



# cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
cfg.SOLVER.CHECKPOINT_PERIOD = 5000
cfg.INPUT.MIN_SIZE_TRAIN = (900,)
cfg.INPUT.MAX_SIZE_TRAIN = 1800

cfg.INPUT.MIN_SIZE_TEST = 900
cfg.INPUT.MAX_SIZE_TEST = 1800
cfg.INPUT.RANDOM_FLIP = "horizontal"



# Type of gradient clipping, currently 2 values are supported:
# # - "value": the absolute values of elements of each gradients are clipped
# # - "norm": the norm of the gradient for each parameter is clipped thus
# #   affecting all elements in the parameter
# cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"
# # Maximum absolute value used for clipping gradients
# cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
# # Floating point number p for L-p norm to be used with the "norm"
# # gradient clipping type; for L-inf, please specify .inf
# cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 5.0



# cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"

# cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
# cfg.SOLVER.WARMUP_ITERS = 1000

cfg.SOLVER.GAMMA = 0.1
cfg.SOLVER.STEPS = (15000, 20500, 25000, 30000, 35000, 40000, 60000, 80000) # [] => do not decay learning rate

# Options: TrainingSampler, RepeatFactorTrainingSampler (class imbalance)
cfg.DATALOADER.SAMPLER_TRAIN = "RepeatFactorTrainingSampler"
# Repeat threshold for RepeatFactorTrainingSampler
cfg.DATALOADER.REPEAT_THRESHOLD = 0.001

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256  #128 # default is 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(unique_labels) 



cfg.OUTPUT_DIR = OUTPUT_DIR
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

trainer = DefaultTrainer(cfg)
val_loss = ValidationLoss(cfg) 
trainer.register_hooks([val_loss])
trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]

joblib.dump(cfg, cfg.OUTPUT_DIR+"/trn_config.bin")

trainer.resume_or_load(resume=True)
trainer.train()


