# MaskR-CNN-Instance Segmentation & Detection & OCR Pipeline 
## Facebook AI Research's (FAIR'S) Created ---> PyTorch is an open source machine learning library ---> new framework is called Detectron2 which is implemented in Pytorch.
 ![plot](https://github.com/AbirKhan96/SampleImages_AI-ML/blob/main/Track_A-Ladybug-1285.jpg) :dart: **Output of Mask R-CNN .**
 
## Code with Description
 - [Folder_Rename.py](https://github.com/AbirKhan96/MaskR-CNN-InstanceSegmentation/blob/main/src/Folder_Rename.py) This code is use to rename the json file with folder name before it.
 - [DataCleaner_Error_find.py](https://github.com/AbirKhan96/MaskR-CNN-InstanceSegmentation/blob/main/src/DataCleaner_Error_find.py) Get the count of Data and list of Assets and if there is any error in polygon.
 - [Base_Training_code.ipynb](https://github.com/AbirKhan96/MaskR-CNN-InstanceSegmentation/blob/main/src/Base_Training_code.ipynb) Training code 
 - [Base_prediction_code.py](https://github.com/AbirKhan96/MaskR-CNN-InstanceSegmentation/blob/main/src/Base_prediction_code.py) consist of prediction pipeline. To run the code on a tree folder.
 - [Base_prediction_code_top_bottom.py](https://github.com/AbirKhan96/MaskR-CNN-InstanceSegmentation/blob/main/src/Base_prediction_code_top_bottom.py) consist of prediction code to get Height of asset. Calculate height with help of Top and bottom Mask (Altitude). We get the top point of Mask.
 - [config.py](https://github.com/AbirKhan96/MaskR-CNN-InstanceSegmentation/blob/main/src/config.py) class TrainConfig, class DataConfig, class ModelConfig.
 - [get_pixel_pcmc.py](https://github.com/AbirKhan96/MaskR-CNN-InstanceSegmentation/blob/main/get_pixel_pcmc.py) Creates a csv file with pixel info.
 - [PCMC_lidar_part.py](https://github.com/AbirKhan96/MaskR-CNN-InstanceSegmentation/blob/main/PCMC_lidar_part.py) Code to generating lidar point and Shape file.
 - [altitude_estimation.py](https://github.com/AbirKhan96/MaskR-CNN-InstanceSegmentation/blob/main/altitude_estimation.py) Run this code after [PCMC_lidar_part.py](https://github.com/AbirKhan96/MaskR-CNN-InstanceSegmentation/blob/main/PCMC_lidar_part.py) to get the height of asset in csv.
 
 ![plot](https://github.com/AbirKhan96/SampleImages_AI-ML/blob/main/Lidar..png) :dart: **Lidar looks like in Fugro viewer.**  
 
 
 ## How To Train

- Annotate images using `labelme` (segmentation)
  
- Clean `.json` files. (Currently using `utisl/checknames.py`)
  
- Paste only the output json files inside `store/data/YourDatasetName/all_jsons/` (create dir if not exists)
  ```
  store
    ├── data
    │   └── ShopHoarding
    │       └── all_jsons
    │           ├── 2.json
    │           ├── 3.json
    │           ├── 5.json
    │           ├── 6.json
    │           ├── 7.json
    │           ├── 8.json
    │           └── 9.json
    └── model
        └── ShopHoardingModel
  ```

- Add data config inside `config.py` -> `class DataConfig`
  ```python
    class DataConfig:
        class ShopHoardingDataset:

            data_type = "InstanceSegmentation"

            all_jsons_dir = str(DATA_BASE / 'ShopHoarding' / 'all_jsons') + '/'
            split_dataset_dir = str(DATA_BASE / 'ShopHoarding') + '/'

            train_test_split = 0.9
            test_hldt_split = 0.4

            to_shape = (7054, 3527)
            ALLOWED_CLASSES = [
                                 'hoardings', 
                                 'garbage_bins', 
                                 'transformer', 
                                 'bus_stops', 
                                 'fire_hydrants', 
                                 'traffic_sign', 
                                 'traffic_signal', 
                                 'strom_water_vent'                              
            ]

            thing_classes = ['hoardings','garbage_bins','transformer', 
                             'bus_stops', 
                             'fire_hydrants', 
                             'traffic_sign', 
                             'traffic_signal', 
                             'strom_water_vent']
            ext = 'JPG'

  ```

- Add model config inside `config.py` -> `class ModelConfig`
  ```python
    class ModelConfig:

      class ShopHoardingModel:
        zoo_path = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        save_dir = str(MODEL_BASE / 'ShopHoardingModel') + '/'
  ```
- Add training config inside `config.py` -> `class TrainConfig`
  ```python
  class TrainConfig:

    max_iter = 50
    base_lr = 0.00025
    batch_size_per_img = 6
    num_workers = 20  # os.cpu_count()
    ims_per_batch = 2
  ```

- **ToDo:** Prepare data config

- Run the below python code inside `src` dir
  ```python
  from pipeline.train.det2.trainer import Det2Trainer
  from config import TrainConfig, DataConfig, ModelConfig

  # configure trainer
  trainer = Det2Trainer(
    data=DataConfig.ShopHoardingDataset,
    model=ModelConfig.ShopHoardingModel,
    cfg=TrainConfig)

  # process the json files into trainable data
  trainer.prepare_data()

  # generates out directory for assets' model after training
  # and evaluating
  trainer.start(
      resume=False,
      train_dataset=("dataset_train",),
      test_dataset=("dataset_test",))
  ```
