import os

from pipeline.utils.file.name import rm_names_with_parenthesis, get_files_of
from pipeline.utils.file.save import dump
from pipeline.utils.json.json_folder_processor import JsonFolderProcessor
from pipeline.utils.file.train_test_split import TrainTestSplit
from pipeline.utils.labelme2coco import labelme2coco
from pipeline.utils.data_dict import DataDictCreator
from pipeline.utils.file.save import dump

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer


class Det2Trainer:

    # ======================================================================================================
    # beg: constructor
    # ======================================================================================================
    def __init__(self, data, model, cfg):
        """
        :param data:
            dataset config that has
                - data_type: instance seg / panoptic seg / semantic seg / detection ...
                - all_jsons_dir: path to all jsons
                - split_dataset_dir: base dir where train, test and holdout is created 
                - train_test_split: 0.8
                - test_hldt_split: 0.8
                - to_shape: shape to which b64 data, annots and images(optionally) are converted
                - ext: new extension of image file names
                - thing_classes: list of all class names in jsons dir
        :param model:
            model config
                - zoo_path: str = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
                - save_dir: output dir path at which trained model is saved
        :param max_iter: (int)
            default `3000`
        :param base_lr: (float)
            starting lr
        :param batch_size_per_img: (int)
            default `512`. Bigger it is, faster the training
        :param num_workers: (int)
            cpu cores to use for training
        :param ims_per_batch: (int)
            default `2`
        """

        self.dta_cfg = data
        self.mdl_cfg = model

        os.makedirs(self.mdl_cfg.save_dir, exist_ok=True)
        dump(self.dta_cfg, self.mdl_cfg.save_dir+'dta_cfg.bin')
        dump(self.mdl_cfg, self.mdl_cfg.save_dir+'mdl_cfg.bin')
        dump(cfg, self.mdl_cfg.save_dir+'trn_cfg.bin')

        #! todo: instead of storing in self, store in `self.trn_cfg`
        #! and make changes in respective places
        self.base_lr = cfg.base_lr if hasattr(cfg, 'base_lr') else 0.00025
        self.max_iter = cfg.max_iter if hasattr(cfg, 'max_iter') else 3000
        self.batch_size_per_img = cfg.batch_size_per_img if hasattr(
            cfg, 'batch_size_per_img') else 512
        self.num_workers = cfg.num_workers if hasattr(
            cfg, 'num_workers') else 2
        self.ims_per_batch = cfg.ims_per_batch if hasattr(
            cfg, 'ims_per_batch') else 2
        self.num_workers = cfg.num_workers if hasattr(
            cfg, 'num_workers') else 1

        print("="*200)
        print('learning param base_lr            :', self.base_lr)
        print('learning param max_iter           :', self.max_iter)
        print('learning param batch_size_per_im  :', self.batch_size_per_img)
        print('learning param ims_per_batch      :', self.ims_per_batch)
        print('learning param num_workers        :', self.num_workers)
        print("="*200)
    # ======================================================================================================
    # end: constructor
    # ======================================================================================================

    # ======================================================================================================
    # beg: rehape annots, b64 data and write images from json
    # ======================================================================================================
    def proc_json_files(self, all_jsons_dir, write_dir, to_shape, ext, pipeline=['write_original_image', 'write_reshaped_b64json']):

        processor = JsonFolderProcessor(
            all_jsons_dir,
            to_shape,
            pipeline=pipeline,
            ext=ext
        )

        processor.write_to_dir(all_jsons_dir)  # overwrite
    # ======================================================================================================
    # end: rehape annots, b64 data and write images from json
    # ======================================================================================================

    # ======================================================================================================
    # beg: train test split
    # ======================================================================================================
    def train_test_split(self, img_and_json_dir, out_dir, ext, ratio=0.8, test_hldt_split=0.8):

        splitter = TrainTestSplit(
            ratio=ratio,
            test_hldt_split=test_hldt_split,
            img_and_json_dir=img_and_json_dir,
            ext=ext)

        splitter.move_to_folders_at(out_dir)
    # ======================================================================================================
    # end: train test split
    # ======================================================================================================

    # ======================================================================================================
    # beg: to coco format
    # ======================================================================================================
    def labeme_to_coco(self, base_dir, list_of_dirs=['train', 'test', 'holdout']):

        for dirname in list_of_dirs:

            dir_path = base_dir + '/' + dirname
            json_files = get_files_of(ext='json', at=dir_path)
            labelme2coco(json_files, save_json_name=base_dir +
                         dirname + '.json')
    # ======================================================================================================
    # end: to coco format
    # ======================================================================================================

    # ======================================================================================================
    # beg: rm dup files. (temporarily using parenthesis)
    # ======================================================================================================
    def rm_dups(self, directory):
        rm_names_with_parenthesis(directory)
    # ======================================================================================================
    # end: rm dup files. (temporarily using parenthesis)
    # ======================================================================================================

    # ======================================================================================================
    # beg: register datasets
    # ======================================================================================================
    def __save_metadata(self, reg_name, at):
        os.makedirs(at, exist_ok=True)
        dump(
            obj=MetadataCatalog.get(reg_name),
            saveby=at+'metadata.bin')

    def reg_datasets(self, thing_classes, to_shape, metadata_save_path,
                     base_dir, dirs=["train", "test", "holdout"], prefix=""):

        obj = DataDictCreator(thing_classes=thing_classes, to_shape=to_shape)
        for d in dirs:

            DatasetCatalog.register(
                prefix + "dataset_" + d, lambda d=d: obj.get_data_dicts(f'{base_dir}/' + d))
            MetadataCatalog.get(prefix + "dataset_" +
                                d).set(thing_classes=obj.THING_CLASSES)

        #train_data_metadata = MetadataCatalog.get("dataset_train")
        #test_data_metadata = MetadataCatalog.get("dataset_test")
        self.__save_metadata(prefix+"dataset_train", at=metadata_save_path)
    # ======================================================================================================
    # end: register datasets
    # ======================================================================================================

    # ======================================================================================================
    # beg: prepare data
    # ======================================================================================================
    def prepare_data(self, dta_prp_cfg):
        """
        use all the methods defined above
        """
        if dta_prp_cfg.proc_json_files is True:
            self.proc_json_files(
                pipeline=['write_reshaped_image', 'write_reshaped_b64json'],
                all_jsons_dir=self.dta_cfg.all_jsons_dir,
                write_dir=self.dta_cfg.all_jsons_dir,
                to_shape=self.dta_cfg.to_shape,
                ext=self.dta_cfg.ext)

        if dta_prp_cfg.rm_dups is True:
            self.rm_dups(directory=self.dta_cfg.all_jsons_dir)

        if dta_prp_cfg.train_test_split is True:
            self.train_test_split(
                ratio=self.dta_cfg.train_test_split,
                test_hldt_split=self.dta_cfg.test_hldt_split,
                img_and_json_dir=self.dta_cfg.all_jsons_dir,
                out_dir=self.dta_cfg.split_dataset_dir,
                ext=self.dta_cfg.ext)

        if dta_prp_cfg.labelme2coco is True:
            self.labeme_to_coco(
                base_dir=self.dta_cfg.split_dataset_dir,
                list_of_dirs=['train', 'test', 'holdout']
            )

        if dta_prp_cfg.reg_datasets is True:
            self.reg_datasets(
                thing_classes=self.dta_cfg.thing_classes,
                to_shape=self.dta_cfg.to_shape,
                base_dir=self.dta_cfg.split_dataset_dir,
                metadata_save_path=self.mdl_cfg.save_dir,
            )
    # ======================================================================================================
    # end: prepare data
    # ======================================================================================================

    # ======================================================================================================
    # beg: start training
    # ======================================================================================================
    def start(self, resume=False, train_dataset=("dataset_train",), test_dataset=("dataset_test",)):

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(self.mdl_cfg.zoo_path))

        # must be registered
        cfg.DATASETS.TRAIN = train_dataset
        cfg.DATASETS.TEST = test_dataset

        cfg.DATALOADER.NUM_WORKERS = self.num_workers
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.mdl_cfg.zoo_path)
        cfg.SOLVER.IMS_PER_BATCH = self.ims_per_batch
        cfg.SOLVER.BASE_LR = self.base_lr
        cfg.SOLVER.MAX_ITER = self.max_iter
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = self.batch_size_per_img
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(self.dta_cfg.thing_classes)

        cfg.OUTPUT_DIR = self.mdl_cfg.save_dir
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume)
        trainer.train()
    # ======================================================================================================
    # end: start training
    # ======================================================================================================
