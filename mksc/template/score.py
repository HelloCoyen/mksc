import configparser
import os

from statsmodels.iolib.smpickle import load_pickle

from mksc.feature_engineering import scoring


def main():
    """
    特征评分卡制作主程序入口
    """
    cfg = configparser.ConfigParser()
    cfg.read(os.path.join(os.getcwd(), 'template/config', 'configuration.ini'), encoding='utf_8_sig')
    odds = cfg.get('SCORECARD', 'ODDS')
    score = cfg.get('SCORECARD', 'SCORE')
    pdo = cfg.get('SCORECARD', 'PDO')
    feature_engineering = load_pickle("result/feature_engineering.pickle")
    woe_result = feature_engineering["woe_result"]
    model = load_pickle("result/lr.pickle")
    coefficient = list(zip(feature_engineering["feature_selected"], list(model.coef_[0])))
    coefficient.append(("intercept_", model.intercept_[0]))
    coefs = dict(coefficient)
    scoring.make_card(coefs, woe_result, odds, score, pdo)


if __name__ == "__main__":
    main()
