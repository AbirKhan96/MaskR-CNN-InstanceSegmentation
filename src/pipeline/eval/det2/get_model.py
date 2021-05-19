import os
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from pipeline.utils.file.save import load


class GetTrained:

    def __init__(
            self,
            model_name,
            base_dir="store/model/",
            zoo_path="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"):

        from_dir = base_dir + model_name
        assert os.path.exists(
            from_dir), f"{from_dir} does not exist!! it is where trained model should exist."

        trn_cfg = load(os.path.join(from_dir, "trn_cfg.bin"))
        dta_cfg = load(os.path.join(from_dir, "dta_cfg.bin"))

        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(zoo_path))
        self.cfg.MODEL.WEIGHTS = os.path.join(from_dir, "model_final.pth")
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = trn_cfg.batch_size_per_img
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(dta_cfg.thing_classes)

    def predictor(self, thresh=0.5, cfg=False):
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresh
        if cfg:
            return DefaultPredictor(self.cfg), self.cfg
        return DefaultPredictor(self.cfg)

    def fetch(self, **kwargs):
        return self.predictor(**kwargs)


class GetPretrained:

    def __init__(self, zoo_path="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(zoo_path))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(zoo_path)

    def predictor(self, thresh=0.5):
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresh
        if cfg:
            return DefaultPredictor(self.cfg), self.cfg
        return DefaultPredictor(self.cfg)

    def fetch(self, **kwargs):
        return self.predictor(**kwargs)


if __name__ == '__main__':

    model = GetTrained("ShopHoardingModel").fetch(thresh=0.5)
    # model = GetPretrained().fetch(thresh=0.5)
