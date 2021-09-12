# MaskR-CNN-Instance Segmentation & Detection & OCR Pipeline 
## Facebook AI Research's (FAIR'S) Created ---> PyTorch is an open source machine learning library ---> new framework is called Detectron2 which is implemented in Pytorch.
 ![plot](https://github.com/AbirKhan96/SampleImages_AI-ML/blob/main/Track_A-Ladybug-1285.jpg) ## Output of Mask R-CNN

 - [Folder_Rename.py](https://github.com/AbirKhan96/MaskR-CNN-InstanceSegmentation/blob/main/src/Folder_Rename.py) This code is use to rename the json file with folder name before it.
 - [DataCleaner_Error_find.py](https://github.com/AbirKhan96/MaskR-CNN-InstanceSegmentation/blob/main/src/DataCleaner_Error_find.py) Get the count of Data and list of Assets and if there is any error in polygon.
 - [Base_Training_code.ipynb](https://github.com/AbirKhan96/MaskR-CNN-InstanceSegmentation/blob/main/src/Base_Training_code.ipynb) Training code 
 - [Base_prediction_code.py](https://github.com/AbirKhan96/MaskR-CNN-InstanceSegmentation/blob/main/src/Base_prediction_code.py) consist of prediction pipeline. To run the code on a tree folder.
 - [Base_prediction_code_top_bottom.py](https://github.com/AbirKhan96/MaskR-CNN-InstanceSegmentation/blob/main/src/Base_prediction_code_top_bottom.py) consist of prediction code to get Height of asset. Calculate height with help of Top and bottom Mask (Altitude). We get the top point of Mask.
 - [config.py](https://github.com/AbirKhan96/MaskR-CNN-InstanceSegmentation/blob/main/src/config.py) class TrainConfig, class DataConfig, class ModelConfig.
 - [get_pixel_pcmc.py](https://github.com/AbirKhan96/MaskR-CNN-InstanceSegmentation/blob/main/get_pixel_pcmc.py) Creates a csv file with pixel info.
 - [PCMC_lidar_part.py](https://github.com/AbirKhan96/MaskR-CNN-InstanceSegmentation/blob/main/PCMC_lidar_part.py) Code to generating lidar point and Shape file.
 - [altitude_estimation.py](https://github.com/AbirKhan96/MaskR-CNN-InstanceSegmentation/blob/main/altitude_estimation.py) Run this code after [PCMC_lidar_part.py](https://github.com/AbirKhan96/MaskR-CNN-InstanceSegmentation/blob/main/PCMC_lidar_part.py) to get the height of asset in csv.
 ![plot](https://github.com/AbirKhan96/SampleImages_AI-ML/blob/main/Lidar..png) Lidar looks like in Fugro viewer.  
