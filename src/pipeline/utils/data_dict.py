import os
import numpy as np
import json
from detectron2.structures import BoxMode


class DataDictCreator:

    def __init__(self, thing_classes, to_shape):

        self.THING_CLASSES = thing_classes
        self.RECORD_W, self.RECORD_H = to_shape

    def get_data_dicts(self, directory):

        classes = self.THING_CLASSES
        dataset_dicts = []

        for idx, filename in enumerate([file for file in os.listdir(directory) if file.endswith('.json')]):
            json_file = os.path.join(directory, filename)
            with open(json_file) as f:
                img_anns = json.load(f)

            record = {}
            filename = os.path.join(directory, img_anns["imagePath"])
            
            record["file_name"] = filename
            record["height"] = self.RECORD_H
            record["width"] = self.RECORD_W
            record['image_id'] = idx
        
            annos = img_anns["shapes"]
            objs = []
            for anno in annos:
                px = [a[0] for a in anno['points']]
                py = [a[1] for a in anno['points']]
                poly = [(x, y) for x, y in zip(px, py)]
                poly = [p for x in poly for p in x]

                obj = {
                    "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [poly],
                    "category_id": classes.index(anno['label']),
                    "iscrowd": 0
                }
                objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
        return dataset_dicts


if __name__ == '__main__':

    THING_CLASSES = ['MicroWave', 'panel_antenna', 'RRU']
    TO_SHAPE = ()

    obj = DataDictCreator(THING_CLASSES, TO_SHAPE)

    # register in DatasetCatalog
    from detectron2.data import DatasetCatalog, MetadataCatalog
    for d in ["train", "test"]:

        DatasetCatalog.register("dataset_" + d, lambda d=d: obj.get_data_dicts('dataset/' + d))
        MetadataCatalog.get("dataset_" + d).set(thing_classes=obj.THING_CLASSES)
    
    train_data_metadata = MetadataCatalog.get("dataset_train")
    test_data_metadata = MetadataCatalog.get("dataset_test")