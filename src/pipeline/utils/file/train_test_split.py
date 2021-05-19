from pipeline.utils.file.name import get_ext
import os, shutil, glob

# ======================================================================================================
# beg: train test split witj (.jpg and .json)
# ======================================================================================================
class TrainTestSplit:

    def __init__(self, ratio, test_hldt_split, img_and_json_dir, ext='JPG'):
        
        self.ext = ext
        self.ratio = ratio
        self.test_hldt_split = test_hldt_split
        self.img_and_json_dir = img_and_json_dir


    @staticmethod
    def get_unique_names(img_and_json_dir, ext):

        unq_json_files = set()
        unq_img_files = set()
        for f in [f for f in os.listdir(img_and_json_dir) if get_ext(f) in ([ext.lower(), 'json'])]:
            if get_ext(f) in ['json']: unq_json_files.add(f.split(".")[-2])
            if get_ext(f) in [ext.lower()]: unq_img_files.add(f.split(".")[-2])
        
        unq_pairs = set()
        for lab in unq_json_files:
            if lab in unq_img_files:
                unq_pairs.add(lab)

        return unq_pairs


    def get_train_test_names(self, ratio: float, test_split_ratio: float):
        
        unq_pairs = list(TrainTestSplit.get_unique_names(self.img_and_json_dir, self.ext))
        
        lim = int(len(unq_pairs)*ratio)        
        train_pairs = unq_pairs[:lim]
        test_and_holdout_pairs = unq_pairs[lim:]

        lim = int(len(test_and_holdout_pairs)*test_split_ratio)
        test_pairs = test_and_holdout_pairs[:lim]
        hldt_pairs = test_and_holdout_pairs[lim:]

        return train_pairs, test_pairs, hldt_pairs


    def move_to_folders_at(self, write_dir):

        os.makedirs(f"{write_dir}/train/", exist_ok=True)
        os.makedirs(f"{write_dir}/test/", exist_ok=True)
        os.makedirs(f"{write_dir}/holdout/", exist_ok=True)
        
        train_pairs, test_pairs, hldt_pairs = self.get_train_test_names(self.ratio, self.test_hldt_split)

        for fname in train_pairs:
            shutil.move(self.img_and_json_dir + fname + ".json", write_dir + "train/" + fname + ".json")
            shutil.move(self.img_and_json_dir + fname + f".{self.ext}", write_dir + "train/" + fname + f".{self.ext}")

        for fname in test_pairs:
            shutil.move(self.img_and_json_dir + fname + ".json", write_dir + "test/" + fname + ".json")
            shutil.move(self.img_and_json_dir + fname + f".{self.ext}", write_dir + "test/" + fname + f".{self.ext}")

        for fname in hldt_pairs:
            shutil.move(self.img_and_json_dir + fname + ".json", write_dir + "holdout/" + fname + ".json")
            shutil.move(self.img_and_json_dir + fname + f".{self.ext}", write_dir + "holdout/" + fname + f".{self.ext}")
# ======================================================================================================
# end: train test split witj (.jpg and .json)
# ======================================================================================================


if __name__ == '__main__':

    jsons_and_images_dir = ".."
    ext = 'JPG'
    out_dir = './'

    splitter = TrainTestSplit(ratio=0.8, test_hldt_split=0.8, img_and_json_dir=jsons_and_images_dir, ext=ext)
    splitter.move_to_folders_at(out_dir)