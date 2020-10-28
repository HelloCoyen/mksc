from statsmodels.iolib.smpickle import load_pickle
from feature_engineering.scoring import process as ps
import configparser
import os

def main():
    """
    评分卡制作主程序入口
    """
    cfg = configparser.ConfigParser()
    cfg.read(os.path.join(os.getcwd(), 'conf', 'config/configuration.ini'), encoding='utf_8_sig')
    odds = cfg.get('configuration', 'odds')
    score = cfg.get('configuration', 'score')
    pdo = cfg.get('configuration', 'pdo')
    woe_result = load_pickle("result/feature_engineering.pickle")["woe_result"]
    coefs = load_pickle("result/coefs.pickle")
    ps.make_card(coefs, woe_result, odds, score, pdo)


if __name__ == "__main__":
    main()
