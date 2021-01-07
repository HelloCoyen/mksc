import numpy as np

def fix_abnormal_value(feature, threshold=0.05, method='boundary'):
    """
    修正数据框中的数值型变量中的异常值

    Args:
        feature: 待修正的数据框
        threshold: 异常值替换阈值
        method: 补值方法
            -- boundary：缩放到边界
            -- drop: 丢弃
    Returns:
        feature: 已处理数据框
        abnormal_value: 异常值统计结果
    """
    numeric_var = feature.select_dtypes(exclude=['object', 'datetime']).columns
    abnormal_value = {'result': {}, 'replace': []}
    for c in numeric_var:
        sm = feature[c].describe()
        iqr = sm['75%'] - sm['25%']
        min_ = sm['25%'] - 1.5*iqr
        max_ = sm['75%'] + 1.5*iqr
        abnormal_value_indexes = list(feature.loc[(feature[c] <= min_) | (feature[c] >= max_)].index)
        abnormal_value_length = len(abnormal_value_indexes)
        abnormal_value_rate = abnormal_value_length/feature[c].count()
        abnormal_value['result'][c] = {'abnormal_value_list': abnormal_value_indexes,
                                       'abnormal_value_length': abnormal_value_length,
                                       'abnormal_value_rate': abnormal_value_rate,
                                       'max': max_,
                                       'min': min_}
        if abnormal_value_rate <= threshold:
            abnormal_value['replace'].append(c)
            if method == "drop":
                feature.loc[:, c] = feature.loc[:, c].apply(lambda x: x if (x < max_) & (x > min_) else np.nan)
            else:
                feature.loc[:, c] = feature.loc[:, c].apply(lambda x: max_ if x > max_ else(min_ if x < min_ else x))
    return feature, abnormal_value
