import argparse
import configparser
import os
from datetime import date
from math import log

import pandas as pd
from statsmodels.iolib.smpickle import load_pickle

import mksc
from custom import Custom
from mksc.feature_engineering import preprocess
from mksc.feature_engineering import scoring
from mksc.feature_engineering import transform


def main(model_name, dataset="apply", card=False, score=False, remote=False, local=False):
    # 数据、模型加载
    model, threshold = load_pickle(f'result/{model_name}.pickle')

    feature_engineering = load_pickle('result/feature_engineering.pickle')
    data = mksc.load_data(dataset, local=local)
    numeric_var, category_var, datetime_var, label_var = preprocess.get_variable_type()
    numeric_var, category_var, datetime_var = [list(set(t) & set(data.columns)) for t in (numeric_var, category_var, datetime_var)]
    feature = data[numeric_var + category_var + datetime_var]
    label = []

    cs = Custom()
    # 自定义数据清洗
    feature, label = cs.clean_data(feature, label)

    # 数据类型转换
    feature[numeric_var] = feature[numeric_var].astype('float')
    feature[category_var] = feature[category_var].astype('object')
    feature[datetime_var] = feature[datetime_var].astype('datetime64')

    # 自定义特征组合模块
    feature = cs.feature_combination(feature)

    feature = feature[feature_engineering['feature_selected']]

    # 数据处理
    feature = transform(feature, feature_engineering)

    # 应用预测
    print(">>> 应用预测")
    res_label = pd.DataFrame(model.predict(feature), columns=['label_predict'])
    res_prob = pd.DataFrame(model.predict_proba(feature), columns=['probability_0', "probability_1"])
    res_prob['res_odds'] = res_prob['probability_0'] / res_prob["probability_1"]
    res_prob['label_threshold'] = res_prob['probability_1'].apply(lambda x: 0 if x < threshold else 1)
    res = pd.concat([data, res_label, res_prob], axis=1)

    if card and model_name == 'lr':
        # 转化评分
        score_card = load_pickle('result/card.pickle')
        res = scoring.transform_score(res, score_card)

    if score:
        print(">>> 概率转换评分")
        cfg = configparser.ConfigParser()
        cfg.read(os.path.join(os.getcwd(), 'config', 'configuration.ini'), encoding='utf_8_sig')
        odds = cfg.get('SCORECARD', 'odds')
        score = cfg.get('SCORECARD', 'score')
        pdo = cfg.get('SCORECARD', 'pdo')
        a, b = scoring.make_score(odds, score, pdo)
        res['score'] = res_prob['res_odds'].apply(lambda x: a + b * log(float(x)))
        if dataset == "all":
            if cs.adjust_bins:
                bins = cs.adjust_bins
            else:
                from mksc.feature_engineering.binning import tree_binning
                bins = tree_binning(res[label_var], res['score'].to_frame())[0]["result"]["score"]
            print(">>> 训练集分组")
            res['level'] = pd.cut(res['score'], bins)
            temp = res.groupby("level", as_index=False).count()
            temp['rate'] = temp[label_var] / feature.shape[0]
            temp = temp[['level', 'rate']]
            print(temp)
            print(res.head())

    # 结果保存
    print(f">>> 结果保存中，保存模式：{remote}")
    res['load_date'] = str(date.today())
    mksc.save_result(res, filename=f"{dataset}_result.csv", remote=remote)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("-m", "--model", type=str, required=True, help=r"模型选择：xgb\lr\svm\rf\gbdt\lgb\dt\自定义")
    args.add_argument("-d", "--dataset", type=str, default="apply", help="预测数据集")
    args.add_argument("-c", "--card", type=bool, default=False, help="是否制作评分卡")
    args.add_argument("-s", "--score", type=bool, default=False, help="是否转换成评分")
    args.add_argument("-r", "--remote", type=bool, default=False,  help="是否保存远程")
    args.add_argument("-l", "--local", type=bool, default=False,  help="是否读取远程")
    accepted = vars(args.parse_args())
    main(model_name=accepted['model'], dataset=accepted['dataset'], card=accepted['card'],
         score=accepted['score'], remote=accepted['remote'], local=accepted['local'])
