from pathlib import Path

DATA_BASE = Path('store') / 'data'
MODEL_BASE = Path('store') / 'model'


class TrainConfig:

    max_iter = 10000
    base_lr =  0.0025 #0.00025
    batch_size_per_img = 256
    num_workers = 25  # os.cpu_count()
    ims_per_batch = 16

    
class DataPreparationConfig:
    # while training all are true
    proc_json_files = False
    rm_dups = False # todo: checks
    train_test_split = False
    labelme2coco = True # needed for kpis only if files train, test, holdout contensts change
    reg_datasets = True # needed for kpis

    
class DataConfig:

    # ======================================================================================================
    # beg: dataset details
    # ======================================================================================================
    class AllowedClassesDataset:

        data_type = "InstanceSegmentation"

        all_jsons_dir = str(DATA_BASE / 'Allowed_Classes' / 'all_jsons') + '/'
        split_dataset_dir = str(DATA_BASE / 'Allowed_Classes') + '/'

        train_test_split = 0.9
        test_hldt_split = 0.4

        to_shape = (9216, 4608)
        ALLOWED_CLASSES = [
                          'Street Light_LI41','Mobile Tower_MT_1','CCTV Camera_PC535',
                          'Sign Board_SB197','Sign Board_SB170','Tree Guard_TG_1','Pot_TG_3',
                          'Tree Guard_TG_4','Traffic Signal Light_TR10','Bollard_BO13',
                          'Garbage bin_GA04','Street Light_LI517','Garbage bin_GA03',
                          'Zebra_Crossing','Electric Distribution Box_EB38',
                          'Street Light_LI516','Bollard_BO67','Street Light_LI240','Tree_Deciduous'
                          ]

        thing_classes = ['Traffic Signal Light_TR46',
                        'Traffic Signal Light_TR06',
                        'Sign Board_SB8001',
                        'Street_Light_LI240_1',
                        'Electric Pole_SB153',
                        'Electric Pole_EL6005',
                        'Street Light_LI617',
                        'Traffic Signal_TR51',
                        'Sign_Board_SB197',
                        'Electronic Display Board_DB05',
                        'Electric Pole_EL11',
                        'Traffic_Signal_Light_TR44',
                        'Sign_Board_SB170',
                        'Street_Light_LI2014_1',
                        'Electric Pole_EL2485',
                        'Electric Distribution Box_EB505',
                        'Sign Board_SB 1004',
                        'Traffic Signal Overhead_TO28',
                        'Street Light_LI2014_1',
                        'Sign Board_SB190',
                        'Sign_Board_SB8001',
                        'Transformer_EL2481',
                        'Zebra Crossing',
                        'Sign Board_SB193',
                        'Sign Board_SB 1003',
                        'Electric Distribution Box_EB110',
                        'Sign Board_SB 1001',
                        'Electric Distribution Box_EB455',
                        'Sign Board_SB172',
                        'Sign Board_SB 1007',
                        'Hand Pump',
                        'Electric Pole_EL8006',
                        'OFC Marker_01',
                        'SignBoard_SB61',
                        'Telephone Distribution Box_EB62',
                        'Electric Distribution Box_EB507',
                        'Sign Board_SB21',
                        'Sign Board_SB 1006',
                        'Traffic Signal_TO28',
                        'Electronic Display Board_DB11',
                        'Electric Distribution Box_EB02',
                        'Manhole_MH02',
                        'Traffic Signal Light_TR44',
                        'Hording_4',
                        'SignBoard_SB09',
                        'Garbage_bin_GA03',
                        'Tree Guard_TG_2',
                        'Sign_Board_SB153',
                        'Hording_5',
                        'Letter Box_LT01',
                        'Street_Light_LI41',
                        'Traffic Signal Light_TR52',
                        'Hording_1',
                        'Traffic Signal_TR45',
                        'Sign Board_SB23',
                        'Street Light_LI649',
                        'Electric Distribution Box_EB96',
                        'Electric Distribution Box_EB500',
                        'Traffic Signal Light_TR12',
                        'SignBoard_SB21',
                        'CCTV Camera_PC500',
                        'Manhole_MH01',
                        'Traffic Signal Overhead_TO24',
                        'Electric Distribution Box_EB221',
                        'BOREWELL',
                        'CCTV Camera_LI8001',
                        'Electric Distribution Box_EB7004',
                        'Electric Distribution Box_EB256',
                        'Sign Board_SB11',
                        'Sign Board_SB196',
                        'Street Light_LI200',
                        'Electric Distribution Box_EB17',
                        'Bollard_BO31',
                        'Hoarding',
                        'Sign Board_SB153',
                        'Street_Light_LI5000',
                        'Sign Board_SB07',
                        'Electric Distribution Box_EB58',
                        'Sign_Board_SB07',
                        'Telephone Distribution Box_EB46',
                        'Electric Distribution Box_EB480',
                        'Rectangular Manhole',
                        'Traffic Signal Overhead_TO34',
                        'Electric Distribution Box_EB08',
                        'Street Light_LI41',
                        'Mobile Tower_MT_1',
                        'CCTV Camera_PC535',
                        'Sign Board_SB197',
                        'Sign Board_SB170',
                        'Tree Guard_TG_1',
                        'Pot_TG_3',
                        'Traffic Signal Light_TR10',
                        'Bollard_BO13',
                        'Garbage bin_GA04',
                        'Street Light_LI517',
                        'Garbage bin_GA03',
                        'Zebra_Crossing',
                        'Bollard_BO67',
                        'Street Light_LI240',
                        'Street Light_LI516',
                        'Tree_Deciduous',
                        'Electric Distribution Box_EB38',
                        'Tree Guard_TG_4']
        ext = 'JPG'
    # ======================================================================================================
    # end: dataset details
    # ======================================================================================================





class ModelConfig:

    # ======================================================================================================
    # beg: model details
    # ======================================================================================================
    class Allowed_ClassesModel:

        zoo_path = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        save_dir = str(MODEL_BASE / 'Allowed_Classes') + '/'
    # ======================================================================================================
    # end: model details
    # ======================================================================================================
