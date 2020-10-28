from statsmodels.iolib.smpickle import load_pickle
import pandas as pd
import os
import configparser
from core.logger import logger

def file(filename):
    if '.csv' in filename:
        data = pd.read_csv(filename)
        return data
    elif '.pickle' in filename:
        data = load_pickle(filename)
        return data
    elif ('.xls' in filename) or ('.xlsx' in filename):
        data = pd.read_excel(filename)
        return data
    else:
        logger.error("Wrong Data Type")

def config():
    cfg = configparser.ConfigParser()
    cfg.read(os.path.join(os.getcwd(), 'config', 'configuration.ini'), encoding='utf_8_sig')
    return cfg
