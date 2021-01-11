import configparser
import os

import pandas as pd
from statsmodels.iolib.smpickle import load_pickle


def read_data_file(filename, **kwargs):
    """
    读取数据文件
    Args:
        filename:
    Returns:
        返回数据
    """
    if '.csv' in filename:
        return pd.read_csv(filename, **kwargs)
    elif '.pickle' in filename or '.pkl' in filename:
        return load_pickle(filename, **kwargs)
    elif ('.xls' in filename) or ('.xlsx' in filename):
        return pd.read_excel(filename, **kwargs)
    elif ".txt" in filename:
        return pd.read_table(filename, **kwargs)
    else:
        raise ValueError("Wrong Data Type, only [csv/pkl/xlsx/xls/txt]")

def get_config():
    """
    读取当前工作目录下的项目配置文件/get_config/configuration.ini
    Returns:
        cfg: 配置参数对象
    """
    cfg = configparser.ConfigParser()
    cfg_path = os.path.join(os.getcwd(), 'config', 'configuration.ini')
    cfg.read(cfg_path, encoding='utf_8_sig')
    return cfg
