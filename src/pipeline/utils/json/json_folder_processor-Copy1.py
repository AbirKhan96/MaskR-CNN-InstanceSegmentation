import os, json, io, base64, cv2, glob
from pprint import pprint
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import labelme
from loguru import logger
from tqdm import tqdm
from pipeline.utils.image.reshape import shrink

class JsonFolderProcessor:
    """
    labelme json has:
        ['version', 'flags', 'shapes', 'imagePath', 'imageData', 'imageHeight', 'imageWidth']
    

    convert jsons to reshaped jsons and put it in write dir
    Need to (in order):

        i.    write original image to write dir
        ii.   reshape `shapes`
        iii.  overwrite data in json and save in write dir
                + reshaped b64
                + reshaped img w, h
                + new img path
    """

    def __init__(self, jsons_dir_path, to_shape_lab, pipeline=['write_original_image', 'write_reshaped_b64json'], ext='JPG'):
        
        self.ext = ext
        self.all_json_path = JsonFolderProcessor\
            .get_all_ext(jsons_dir_path, ['json'])
        
        # for labels only
        self.tow = to_shape_lab[0]
        self.toh = to_shape_lab[1]
        
        # for writer
        self.pipeline = pipeline


    @staticmethod
    def get_all_ext(path, allowed_ext):
        names = [ f for f in os.listdir(path) if f.split('.')[-1].lower() in allowed_ext ]
        with_base_path = [ (path + f) for f in names ]
        return with_base_path

    @staticmethod
    def get_json_dict(labelme_json_path):
        return json.load(open(labelme_json_path))

    @staticmethod
    def get_img_from_b64(b64_data):
        #base64_decoded = base64.b64decode(b64_data)
        #image = Image.open(io.BytesIO(base64_decoded))
        #image_np = np.array(image)
        return labelme.utils.img_b64_to_arr(b64_data)

    @staticmethod
    def get_filename_from_path(path):
        with_ext = path.split("/")[-1]
        rm_ext = with_ext.split('.')[-2]
        return rm_ext
    # =================================================================================================================
    # beg: writer
    # =================================================================================================================
    def write_original_image(self, json_file_path, file_name):
        """ 
        + create jpg from json file 
        + heavily dependant on `self.single_json_to_img`
        """

        orgnl_img = self.single_json_to_img(json_file_path)        
        #logger.debug(f"writing {orgnl_img.shape} dim image to {self.write_dir + f'{file_name}.{self.ext}'}")
        cv2.imwrite(self.write_dir + f"{file_name}.{self.ext}", orgnl_img)


    def write_reshaped_image(self, json_file_path, file_name):
        """ 
        + create jpg from json file 
        + heavily dependant on `self.single_json_to_img`
        """
        
        orgnl_img = self.single_json_to_img(json_file_path)
        #logger.debug(f"read {orgnl_img.shape} dim image: {file_name}.jpg")

        reshp_img = shrink(orgnl_img, (self.tow, self.toh))
        cv2.imwrite(self.write_dir + f"{file_name}.{self.ext}", reshp_img)
        #logger.debug(f"reshape to {reshp_img.shape} and write at: {self.write_dir + f'{file_name}.{self.ext}'}")
    

    def write_reshaped_b64json(self, json_file_path, file_name):
        """ 
        + change dims of b64 data 
        + heavily dependant on `self.reshape_single_json_annot`
        """

        new_json = self.reshape_single_json_annot(json_file_path) 
        #logger.debug(f"writing new reshaped b64 in json to {self.write_dir + f'{file_name}.json'}")
        with open(self.write_dir + f"{file_name}.json", 'w', encoding='utf-8') as f:
            json.dump(new_json, f, ensure_ascii=False, indent=4)


    def write_to_dir(self, write_dir):
        """ use methods defined in `self.pipeline` """
        self.write_dir = write_dir
        os.makedirs(self.write_dir, exist_ok=True)

        for json_file_path in tqdm(self.all_json_path):
            # write original image to write dir
            file_name = JsonFolderProcessor.get_filename_from_path(json_file_path)
            try:
                for write_proc in self.pipeline:
                    getattr(self, write_proc)(json_file_path, file_name)
            except Exception as e:
                logger.error(f"{file_name} could not be processed!!")
    # =================================================================================================================
    # beg: writer
    # =================================================================================================================


    # =================================================================================================================
    # beg: json to img
    # =================================================================================================================
    def single_json_to_img(self, json_path):
        dict_ = JsonFolderProcessor.get_json_dict(json_path)
        imgnp = JsonFolderProcessor.get_img_from_b64(dict_['imageData'])
        return imgnp
    # =================================================================================================================
    # end: json to img
    # =================================================================================================================


    # =================================================================================================================
    # beg: json to reshaped json
    # =================================================================================================================
    def reshape_single_json_annot(self, labelme_json_path):

        dic = JsonFolderProcessor.get_json_dict(labelme_json_path)

        wratio = self.tow / dic['imageWidth']
        hratio = self.toh / dic['imageHeight']
        new_wh = (self.tow, self.toh)

        new_shapes = JsonFolderProcessor.get_resized_shapes(dic['shapes'], wratio, hratio)
        new_path = JsonFolderProcessor.get_new_path(dic['imagePath'], self.write_dir, self.ext)
        new_b64 = JsonFolderProcessor.get_resized_b64(dic['imageData'], (self.tow, self.toh) ,(wratio, hratio))
        
        dic['shapes'] = new_shapes
        dic['imagePath'] = new_path
        dic['imageWidth'] = int(new_wh[0])
        dic['imageHeight'] = int(new_wh[1])
        dic['imageData'] = new_b64
        dic['imageHeightRatio'] = hratio
        dic['imageWidthRatio'] = wratio

        return dic

    # return resized b64
    @staticmethod
    def get_resized_b64(old_b64, to_dim, ratio):
        im = JsonFolderProcessor.get_img_from_b64(old_b64)
        
        if min(ratio) < 1: interpol = cv2.INTER_LINEAR
        else: interpol = cv2.INTER_AREA

        im = cv2.resize(im, to_dim, interpolation = interpol)
        #return base64.b64encode(im)
        return labelme.utils.img_arr_to_b64(im).decode()

    # return new path
    @staticmethod
    def get_new_path(old_path, write_dir, ext):
        #return write_dir + old_path.split('/')[-1]
        fname = old_path.split('/')[-1].split('.')[0]
        return fname + "." + ext

    # return new shapes
    @staticmethod
    def get_reshaped_wh_points(old_points, w_ratio, h_ratio):
        new_points = []
        for point in old_points:
            new_points.append([
                point[0] * w_ratio,
                point[1] * h_ratio
            ])
        return new_points

    @staticmethod
    def get_resized_shapes(old_shapes, w_ratio, h_ratio):
        """
        format of `shapes`
            
            [
                {
                    'label': 'any class name',
                    'group_id': null,
                    'shape_type': 'polygon',
                    'flags': {}
                    'points': [
                        [wcood, hcood],
                        [wcood, hcood],
                        [wcood, hcood],
                        ...
                    ]
                },
                {
                    'label': 'any class name',
                    'group_id': null,
                    'shape_type': 'polygon',
                    'flags': {}
                    'points': [
                        [wcood, hcood],
                        [wcood, hcood],
                        [wcood, hcood],
                        ...
                    ]
                },
                ...
                ...
            ]
        """
        
        new_shapes = []
        for any_cls_dic in old_shapes:
            __temp = {}
            __temp['label']      = any_cls_dic['label']
            __temp['group_id']   = any_cls_dic['group_id']
            __temp['shape_type'] = any_cls_dic['shape_type']
            __temp['flags']      = any_cls_dic['flags']
            __temp['points']     = JsonFolderProcessor.get_reshaped_wh_points(
                                        any_cls_dic['points'], w_ratio, h_ratio)
            new_shapes.append(__temp)
        return new_shapes
    # =================================================================================================================
    # end: json to reshaped json
    # =================================================================================================================


    if __name__ == '__main__':
        
        # reshapes annotations but does not reshape images
        JSONS_DIR_PATH = "antenna_jsons/"
        OUT_PATH = JSONS_DIR_PATH

        TO_SHAPE =  (5472, 3648)#(800, 500)#(9216, 4608)

        folder_to_process = ''
        jsons_dir_path = JSONS_DIR_PATH + folder_to_process
        write_dir = OUT_PATH + folder_to_process

        processor = JsonFolderProcessor(jsons_dir_path, TO_SHAPE)
        processor.wrie_to_dir(write_dir)