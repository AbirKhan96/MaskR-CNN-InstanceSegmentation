import joblib
from loguru import logger

def dump(obj, saveby='./temp.bin'):
    ret = joblib.dump(obj, saveby)
    logger.debug(f"saved {ret}")

def load(from_path):
    return joblib.load(from_path)