import numpy as np
from mksc.feature_engineering import binning

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

    # woe转化
    woe_result = feature_engineering['woe_result']
    bin_result = feature_engineering['bin_result']
    feature = binning.woe_transform(feature, woe_result, bin_result)

    return feature
