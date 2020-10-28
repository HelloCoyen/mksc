
def fix_missing_value(feature, threshold=0.05):
    """
    修正数据框中的数值型变量中的缺失值

    Args:
        feature: 待修正的数据框
        threshold: 缺失值替换阈值

    Returns:
        feature: 已处理数据框
        missing_filling: 缺失值统计结果
    """
    numeric_var = feature.select_dtypes(exclude=['object', 'datetime']).columns
    missing_filling = {'result': {}, 'replace': []}
    for c in feature:
        missing_rate = feature[c].isna().sum()/len(feature)
        if missing_rate <= threshold and missing_rate:
            missing_filling['result'][c] = {'fill_number': feature[c].mean(), 'missing_value_rate': missing_rate}
            missing_filling['replace'].append(c)
            if c in numeric_var:
                feature[c].fillna(missing_filling['result'][c], inplace=True)
            else:
                feature[c].fillna(feature[c].mode(), inplace=True)
    return feature, missing_filling
