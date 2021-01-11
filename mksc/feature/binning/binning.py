from copy import deepcopy
from math import log

import pandas as pd
from sklearn.tree import DecisionTreeClassifier


def tree_binning(label, feature):
    """
    通过决策树执行模型分箱函数
     y_i: 第i组中1的个数
     y_T: 所有1的个数
     n_i: 第i组中0的个数
     n_T: 所有0的个数
     WOE_i = ln(P_yi/P_ni) = ln(y_i*n_T/y_T/n_i)
     IV = sum(IV_i) = sum((P_yi-P_ni)*WOE_i) = sum((y_i/y_T-n_i/n_T)*WOE_i)

    Args:
        label: 待分箱特征的标签
        feature: 待分箱特征的数据框

    Returns:
        bin_result: 分箱结果数据
        iv_result: IV值统计结果
        woe_result: WOE值统计结果
        woe_adjust_result: 需要调整分箱的结果
    """
    label_name = label.name
    y_t = label[label.values == 1].count()
    n_t = label[label.values == 0].count()
    bin_result = {'result': {}, 'error': []}
    iv_result = {}
    woe_result = {}
    for c in feature.columns:
        temp = pd.concat([feature[c], label], axis=1)
        # 统计空值分布
        binning_data_na = temp[temp[c].isna()]
        if len(binning_data_na):
            y_na = binning_data_na[label_name][binning_data_na[label_name].values == 1].count()
            n_na = binning_data_na[label_name][binning_data_na[label_name].values == 0].count()
            if y_na and n_na:
                df_na = pd.DataFrame([['nan', y_na, n_na]], columns=[c, 'y_i', 'n_i'])
            else:
                bin_result['error'].append(c)
                continue
        else:
            df_na = pd.DataFrame(columns=[c, 'y_i', 'n_i'])

        # cart树确定分箱边界
        binning_data = temp.dropna(subset=[c])
        model = DecisionTreeClassifier(min_samples_leaf=0.05, min_samples_split=0.02, max_depth=5, max_leaf_nodes=15)
        model.fit(binning_data[c].values.reshape(-1, 1), binning_data[label_name].values)
        boundary = model.tree_.threshold
        boundary = sorted(list(boundary[boundary != -2]))
        boundary = [float('-inf')] + boundary + [float('inf')]
        bin_result['result'][c] = boundary

        # 统计分箱结果
        df = pd.merge(pd.cut(temp[c], boundary), temp[label_name], right_index=True, left_index=True)
        df['count'] = 1
        df = df.groupby([c, label_name]).count().unstack(level=1)
        df.reset_index(inplace=True)
        df.columns = [x[0] + str(x[1]) for x in df.columns.ravel()]
        df.rename(columns={'count0': 'n_i', 'count1': 'y_i'}, inplace=True)
        df = pd.concat([df_na, df])
        df.reset_index(inplace=True)
        df['woe_i'] = df.apply(lambda x: log(x['y_i']) + log(n_t) - log(y_t) - log(x['n_i']), axis=1)
        df['iv_i'] = df.apply(lambda x: (x['y_i']/y_t-x['n_i']/n_t)*x['woe_i'], axis=1)
        woe_result[c] = df[[c, 'woe_i']]
        iv_result[c] = df['iv_i'].sum()

    # 分箱调整：筛选出分箱中存在一类的特征，转化为规则预测
    woe_adjust_result = []
    for w in deepcopy(woe_result):
        if woe_result[w].woe_i.isna().any():
            woe_adjust_result.append(w)
            woe_result.pop(w)

    return bin_result, iv_result, woe_result, woe_adjust_result
