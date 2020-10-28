from statsmodels.iolib.smpickle import load_pickle
from feature_engineering import preprocess
from feature_engineering import binning
from feature_engineering import scoring
import numpy as np
import pandas as pd
from .custom import Custom
import mksc

def transform(feature, feature_engineering):

    # 自定义特征组合模块
    feature = Custom.feature_combination(feature)

    # 极端值处理
    abnormal_value = feature_engineering['abnormal_value']
    for c in set(feature.columns) & set(abnormal_value['replace']):
        max_ = abnormal_value['result'][c]['max']
        min_ = abnormal_value['result'][c]['min']
        feature.loc[:, c] = feature.loc[:, c].apply(lambda x: x if (x < max_) & (x > min_) else np.nan)

    # 缺失值处理
    missing_filling = feature_engineering['missing_filling']
    for c in set(feature.columns) & set(missing_filling['replace']):
        fill_number = abnormal_value['result'][c]['fill_number']
        feature[c].fillna(fill_number, inplace=True)

    # woe转化
    woe_result = feature_engineering['woe_result']
    bin_result = feature_engineering['bin_result']
    feature = binning.woe_transform(feature, woe_result, bin_result)

    return feature

def main():

    # 数据、模型加载
    model = load_pickle('result/model.pickle')
    coefs = load_pickle('result/coefs.pickle')
    feature_engineering = load_pickle('result/feature_engineering.pickle')

    data = mksc.load_data()
    numeric_var, category_var, datetime_var, label_var = preprocess.get_variable_type()
    feature = data[numeric_var + category_var + datetime_var]
    label = None

    # 自定义数据清洗
    feature, label = Custom.clean_data(feature, label)

    # 数据类型转换
    feature[numeric_var] = feature[numeric_var].astype('float')
    feature[category_var] = feature[category_var].astype('object')
    feature[datetime_var] = feature[datetime_var].astype('datetime64')
    feature = data[coefs.keys]
    feature.drop(columns=['intercept_'], inplace=True)

    # 数据处理
    feature = transform(feature, feature_engineering)
    res = pd.DataFrame(model.predict(feature), columns=['label'])
    res = pd.concat([feature, res], axis=1)

    # 转化评分
    score_card = ''
    res = scoring.transform_score(res, score_card)
    # 结果保存
    res.save()


if __name__ == '__main__':
    main()
