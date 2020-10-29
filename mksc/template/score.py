from statsmodels.iolib.smpickle import load_pickle
from mksc.feature_engineering import scoring
import configparser
import os

def main():
    """
    评分卡制作主程序入口
    """
    cfg = configparser.ConfigParser()
    cfg.read(os.path.join(os.getcwd(), 'config', 'configuration.ini'), encoding='utf_8_sig')
    odds = cfg.get('SCORECARD', 'odds')
    score = cfg.get('SCORECARD', 'score')
    pdo = cfg.get('SCORECARD', 'pdo')
    woe_result = load_pickle("result/feature_engineering.pickle")["woe_result"]
    coefs = load_pickle("result/coefs.pickle")
    ps.make_card(coefs, woe_result, odds, score, pdo)


if __name__ == "__main__":
    main()
