import configparser
import os

import pandas as pd
from statsmodels.iolib.smpickle import load_pickle


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
        print("Wrong Data Type")

def config():
    """
    读取项目配置文件/config/configuration.ini
    Returns:
        cfg: 配置参数对象
    """
    cfg = configparser.ConfigParser()
    cfg.read(os.path.join(os.getcwd(), 'config', 'configuration.ini'), encoding='utf_8_sig')
    return cfg
