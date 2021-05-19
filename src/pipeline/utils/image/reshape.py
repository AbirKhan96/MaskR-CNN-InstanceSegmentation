import cv2
from pipeline.utils.file.name import get_ext


def shrink(im, to_dim):
    return cv2.resize(im, to_dim, interpolation = cv2.INTER_AREA) 

def eff_shrink(im, to_dim):
    """ slower, efficient """
    return cv2.resize(im, to_dim, interpolation = cv2.INTER_CUBIC) 

def zoom(im, to_dim):
    """ default in cv2.resize """
    return cv2.resize(im, to_dim, interpolation = cv2.INTER_LINEAR) 


def reshape_images(train_test_dir, to_shape, ext='JPG'):

    for dir_ in ['train', 'test']:
        for file in os.listdir(train_test_dir):
            if get_ext(file) in [ext.lower()]:
                im = cv2.imread(train_test_dir + dir_ + "/" + file)
                im = shrink(im, TO_SHAPE) 
                cv2.imwrite(train_test_dir + dir_ + "/" + file, im)