import argparse
import configparser
import os
from datetime import date
from math import log
import pandas as pd
from statsmodels.iolib.smpickle import load_pickle
from mksc.core.saver import save_result
from mksc.utils import load_data, get_variable_type
from mksc.feature.binning import tree_binning
from mksc.feature import transform
from mksc.score.card import make_score
from custom import Custom

def main(model_name, mode, read_local, score=False, save_remote=False):
    # 数据、模型加载
    model, threshold = load_pickle(f'result/{model_name}.pickle')

    feature_engineering = load_pickle('result/feature_engineering.pickle')
    data = load_data(mode, read_local=read_local)
    numeric_var, category_var, datetime_var, label_var = get_variable_type()
    numeric_var, category_var, datetime_var = [list(set(t) & set(data.columns)) for t in (numeric_var, category_var, datetime_var)]
    feature = data[numeric_var + category_var + datetime_var]
    label = [] if mode != "train" else data[label_var]

    cs = Custom()
    # 自定义数据清洗
    feature, label = cs.clean_data(feature, label)

    # 数据类型转换
    feature[numeric_var] = feature[numeric_var].astype('float')
    feature[category_var] = feature[category_var].astype('object')
    feature[datetime_var] = feature[datetime_var].astype('datetime64')

    # 自定义特征组合模块
    feature = cs.feature_combination(feature)

    # 数据处理
    feature = feature[feature_engineering['feature_selected']]
    feature = transform(feature, feature_engineering)

    # 应用预测
    print(">>> 应用预测")
    res_label = pd.DataFrame(model.predict(feature), columns=['label_predict'])
    res_prob = pd.DataFrame(model.predict_proba(feature), columns=['probability_0', "probability_1"])
    res_prob['res_odds'] = res_prob['probability_0'] / res_prob["probability_1"]
    res_prob['label_threshold'] = res_prob['probability_1'].apply(lambda x: 0 if x < threshold else 1)
    res = pd.concat([data, res_label, res_prob], axis=1)

    if score:
        print(">>> 概率转换评分")
        cfg = configparser.ConfigParser()
        cfg.read(os.path.join(os.getcwd(), 'config', 'configuration.ini'), encoding='utf_8_sig')
        odds = cfg.get('SCORECARD', 'odds')
        score = cfg.get('SCORECARD', 'score')
        pdo = cfg.get('SCORECARD', 'pdo')
        a, b = make_score(odds, score, pdo)
        res['score'] = res_prob['res_odds'].apply(lambda x: a + b * log(float(x)))
        bins = tree_binning(res[label_var], res['score'].to_frame())[0]["result"]["score"] if mode == "train" else cs.adjust_bins
        if bins:
            print(">>> 数据集分组")
            res['level'] = pd.cut(res['score'], bins)
            temp = res.groupby("level", as_index=False).count()
            temp['rate'] = temp['label_threshold'] / feature.shape[0]
            temp = temp[['level', 'rate']]
            print(temp)
            print(res.head())

    # 结果保存
    print(f">>> 结果保存中，保存模式：{save_remote}")
    res['load_date'] = str(date.today())
    save_result(res, filename=f"{mode}_result.csv", remote=save_remote)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--mode", type=str, required=True, help="预测模式")
    args.add_argument("-m", "--model", type=str, required=True, help=r"模型选择：xgb\lr\svm\rf\gbdt\lgb\dt\自定义")
    args.add_argument("--score", action="store_true", help="有该参数表示概率转换成得分")
    args.add_argument("--save_remote", action="store_true", help="有该参数表示保存远程")
    args.add_argument("--read_local", action="store_true",  help="有该参数表示读取本地")
    accepted = vars(args.parse_args())
    main(model_name=accepted['model'], mode=accepted['mode'],
         score=accepted['score'], save_remote=accepted['save_remote'], read_local=accepted['read_local'])

