import pickle
from mksc.feature_engineering import seletction
from mksc.feature_engineering import values
from mksc.feature_engineering import binning

class FeatureEngineering(object):
    def __init__(self, feature, label, missing_threshold=(0.9, 0.05), distinct_threshold=0.9, unique_threshold=0.9,
                 abnormal_threshold=0.05, correlation_threshold=0.7):
        self.feature = feature
        self.label = label
        self.threshold = {}
        self.missing_threshold = missing_threshold
        self.distinct_threshold = distinct_threshold
        self.unique_threshold = unique_threshold
        self.abnormal_threshold = abnormal_threshold
        self.correlation_threshold = correlation_threshold

    def run(self):
        """
        特征工程过程函数,阈值参数可以自定义修改
        1. 特征组合
        2. 基于统计特性特征选择：缺失率、唯一率、众数比例
        3. 极端值处理
        4. TODO 异常值处理
        5. 缺失值处理
        6. COMMENT 归一化处理
        7. COMMENT 正态化处理
        8. 最优分箱
        9. IV筛选
        10. TODO PSI筛选
        11. 相关性筛选
        12. woe转化
        13. 逐步回归筛选

        Returns:
            feature： 已完成特征工程的数据框
        """
        feature = self.feature
        label = self.label
        # 基于缺失率、唯一率、众数比例统计特征筛选
        missing_value = seletction.get_missing_value(feature, self.missing_threshold[0])
        distinct_value = seletction.get_distinct_value(feature, self.distinct_threshold)
        unique_value = seletction.get_unique_value(feature, self.unique_threshold)
        feature.drop(set(missing_value['drop'] + distinct_value['drop'] + unique_value['drop']), axis=1, inplace=True)

        # 极端值处理
        feature, abnormal_value = values.fix_abnormal_value(feature, self.abnormal_threshold)

        # 缺失值处理
        feature, missing_filling = values.fix_missing_value(feature, self.missing_threshold[1])

        # 归一化处理
        # feature = pp.fix_scaling(feature)

        # 正态化处理
        # feature, standard_lambda = pp.fix_standard(feature)

        # 数值特征最优分箱，未处理的变量，暂时退出模型
        bin_result, iv_result, woe_result, woe_adjust_result = binning.tree_binning(label, feature)
        bin_error_drop = bin_result['error'] + woe_adjust_result

        # IV筛选
        iv_drop = list(filter(lambda x: iv_result[x] < 0.02, iv_result))
        feature.drop(iv_drop + bin_error_drop, inplace=True, axis=1)

        # 相关性筛选
        cor_drop = seletction.get_cor_drop(feature, iv_result, self.correlation_threshold)
        feature.drop(cor_drop, inplace=True, axis=1)

        # woe转化
        feature = binning.woe_transform(feature, woe_result, bin_result)

        # 逐步回归筛选
        feature_selected = seletction.stepwise_selection(feature, label)
        feature = feature[feature_selected]

        # 中间结果保存
        result = {"missing_value": missing_value,
                  "distinct_value": distinct_value,
                  "unique_value": unique_value,
                  "abnormal_value": abnormal_value,
                  "missing_filling": missing_filling,
                  "bin_result": bin_result,
                  "iv_result": iv_result,
                  "woe_result": woe_result,
                  "woe_adjust_result": woe_adjust_result,
                  "bin_error_drop": bin_error_drop,
                  "iv_drop": iv_drop,
                  "cor_drop": cor_drop,
                  "feature_selected": feature_selected
                  }
        with open('result/feature_engineering.pickle', 'wb') as f:
            f.write(pickle.dumps(result))
        return feature
