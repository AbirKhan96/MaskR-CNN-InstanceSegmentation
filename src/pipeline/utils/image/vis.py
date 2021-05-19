import cv2, random
from loguru import logger
from detectron2.utils.visualizer import Visualizer
from pipeline.train.det2.dataset.data_dict import DataDictCreator

def vis_data_dicts(dataset_dicts, metadata, sample_size=None, display=False, save_dir=False):

    SAMPLES_TO_VIS = sample_size if sample_size is not None else len(dataset_dicts)
    for i, d in enumerate(random.sample(dataset_dicts, SAMPLES_TO_VIS)):
        
        img = cv2.imread(d["file_name"])
        logger.debug(f"{img.shape} read from {d["file_name"]}")
        
        v = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1)
        v = v.draw_dataset_dict(d)

        if display:
            cv2.imshow(d["file_name"], v.get_image()[:, :, ::-1])

        if save_dir:
            cv2.imwrite(f"{save_dir}/segmentation{i}.png",v.get_image()[:, :, ::-1])