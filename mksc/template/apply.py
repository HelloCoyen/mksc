from statsmodels.iolib.smpickle import load_pickle
import pandas as pd
import configparser
from datetime import date
from math import log
import os
import mksc
from mksc.feature_engineering import preprocess
from mksc.feature_engineering import scoring
from mksc.feature_engineering import transform
from custom import Custom

def main(model_name, card=False, score=True):
    # 数据、模型加载
    model, threshold = load_pickle(f'result/{model_name}.pickle')

    feature_engineering = load_pickle('result/feature_engineering.pickle')
    data = mksc.load_data("apply")
    numeric_var, category_var, datetime_var, label_var = preprocess.get_variable_type()
    feature = data[numeric_var + category_var + datetime_var]
    label = []

    # 自定义数据清洗
    feature, label = Custom.clean_data(feature, label)

    # 数据类型转换
    feature[numeric_var] = feature[numeric_var].astype('float')
    feature[category_var] = feature[category_var].astype('object')
    feature[datetime_var] = feature[datetime_var].astype('datetime64')

    # 自定义特征组合模块
    feature = Custom.feature_combination(feature)

    # One-Hot编码
    category_var = feature.select_dtypes(include=['object']).columns
    feature[category_var].fillna("NA", inplace=True)
    if not feature[category_var].empty:
        feature = pd.concat([feature, pd.get_dummies(feature[category_var])], axis=1)
    feature.drop(category_var, axis=1, inplace=True)
    feature = feature[feature_engineering['feature_selected']]

    # 数据处理
    feature = transform(feature, feature_engineering)

    # 应用预测
    res_label = pd.DataFrame(model.predict(feature), columns=['label_predict'])
    res_prob = pd.DataFrame(model.predict_proba(feature), columns=['probability_0', "probability_1"])
    res_prob['res_odds'] = res_prob['probability_0'] / res_prob["probability_1"]
    res_prob['label_threshold'] = res_prob['probability_1'].apply(lambda x: 0 if x < threshold else 1)
    res = pd.concat([data, feature, res_label, res_prob], axis=1)

    if card and model_name == 'lr':
        # 转化评分
        score_card = load_pickle('result/card.pickle')
        res = scoring.transform_score(res, score_card)

    if score:
        cfg = configparser.ConfigParser()
        cfg.read(os.path.join(os.getcwd(), 'config', 'configuration.ini'), encoding='utf_8_sig')
        odds = cfg.get('SCORECARD', 'odds')
        score = cfg.get('SCORECARD', 'score')
        pdo = cfg.get('SCORECARD', 'pdo')
        a, b = scoring.make_score(odds, score, pdo)
        res['Score'] = res_prob['res_odds'].apply(lambda x: a + b * log(float(x)))

    # 结果保存
    res['load_date'] = str(date.today())
    mksc.save_result(res, mode="remote")
    mksc.save_result(res, "apply_result.csv")


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 2:
        main(sys.argv[1])
