import numpy as np
from mksc.feature_engineering import binning
from math import log
from mksc.feature_engineering import values

def transform(feature, feature_engineering):

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

    # 正态化处理
    standard_lambda = feature_engineering['standard_lambda']
    for c in set(feature.columns) & set(standard_lambda.keys()):
        _lambda = standard_lambda[c]
        feature.loc[:, c] = feature.loc[:, c] + 0.5
        feature.loc[:, c] = feature.loc[:, c].apply(lambda x: (x**_lambda - 1) / _lambda if _lambda > 0 else log(x))

    # 归一化处理
    scale_result = feature_engineering['scale_result']
    for c in set(feature.columns) & set(scale_result.keys()):
        mean = scale_result[c]['mean']
        std = scale_result[c]['std']
        feature.loc[:, c] = feature.loc[:, c].apply(lambda x: (x - mean)/std if x else x)

    # woe转化
    woe_result = feature_engineering['woe_result']
    bin_result = feature_engineering['bin_result']
    feature = binning.woe_transform(feature, woe_result, bin_result)

    return feature
