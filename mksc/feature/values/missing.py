
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
            if c in numeric_var:
                fill_value = feature[c].mean()
            else:
                fill_value = feature[c].mode()
            feature[c].fillna(fill_value, inplace=True)
            missing_filling['result'][c] = {'fill_value': fill_value, 'missing_value_rate': missing_rate}
            missing_filling['replace'].append(c)
    return feature, missing_filling
