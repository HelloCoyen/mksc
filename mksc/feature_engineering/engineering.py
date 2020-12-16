import pickle
import pandas as pd
from mksc.feature_engineering import seletction
from mksc.feature_engineering import values
from mksc.feature_engineering import binning
from sklearn.decomposition import PCA

class FeatureEngineering(object):

    def __init__(self, feature, label,
                 missing_threshold=(0.9, 0.05), distinct_threshold=0.9, unique_threshold=0.9,
                 abnormal_threshold=0.05, correlation_threshold=0.7, variance_threshold=0.05):
        self.feature = feature
        self.label = label
        self.missing_threshold = missing_threshold
        self.distinct_threshold = distinct_threshold
        self.unique_threshold = unique_threshold
        self.abnormal_threshold = abnormal_threshold
        self.correlation_threshold = correlation_threshold
        self.variance_threshold = variance_threshold
        self.threshold = {"missing_threshold": self.missing_threshold,
                          "distinct_threshold": self.distinct_threshold,
                          "unique_threshold": self.unique_threshold,
                          "abnormal_threshold": self.abnormal_threshold,
                          "correlation_threshold": self.correlation_threshold}
        print(self.threshold)

    def run(self):
        """
        特征工程过程函数,阈值参数可以自定义修改
        1. 基于统计特性特征选择：缺失率、唯一率、众数比例
        2. 缺失值处理
        TODO 异常值处理
        4. 极端值处理
        5. 正态化处理
        6. 归一化处理
        7. 最优分箱
        8. IV筛选
        TODO PSI筛选
        9. 相关性筛选
        10. woe转化
        11. One-Hot
        12. TODO 降维

        Returns:
            feature: 已完成特征工程的数据框
            label: 已完成特征工程的标签列
        """
        feature = self.feature.copy()
        label = self.label.copy()

        # 单变量筛选：基于缺失率、唯一率、众数比例统计特征筛选
        print("单变量Filter: 缺失值过滤")
        missing_value = seletction.get_missing_value(feature, self.missing_threshold[0])
        print("单变量Filter: 唯一率过滤")
        distinct_value = seletction.get_distinct_value(feature, self.distinct_threshold)
        print("单变量Filter: 众数比率过滤")
        unique_value = seletction.get_unique_value(feature, self.unique_threshold)
        feature.drop(set(missing_value['drop'] + distinct_value['drop'] + unique_value['drop']), axis=1, inplace=True)

        # 缺失值处理
        print("单变量预处理: 缺失值补充")
        feature, missing_filling = values.fix_missing_value(feature, self.missing_threshold[1])

        # 极端值处理
        print("单变量预处理: 极端值值修正")
        feature, abnormal_value = values.fix_abnormal_value(feature, self.abnormal_threshold)

        # 标准化处理
        print("单变量预处理: 标准归一化")
        feature, scale_result = values.fix_scaling(feature, self.variance_threshold)
        feature.drop(scale_result['drop'], axis=1, inplace=True)

        # 正态化处理
        # feature, standard_lambda = values.fix_standard(feature)

        # 数值特征最优分箱，未处理的变量，暂时退出模型
        print("最优分箱")
        bin_result, iv_result, woe_result, woe_adjust_result = binning.tree_binning(label, feature)
        bin_error_drop = bin_result['error'] + woe_adjust_result
        print(f"空值规则预测：{bin_result['error']}")
        print(f"非空值规则预测：{woe_adjust_result}")

        # IV筛选
        print("单变量Filter: IV值过滤")
        iv_drop = list(filter(lambda x: iv_result[x] < 0.02, iv_result))
        feature.drop(iv_drop + bin_error_drop, inplace=True, axis=1)

        # 相关性筛选
        print("多变量Filter: 相关系数过滤")
        cor_drop = seletction.get_cor_drop(feature, iv_result, self.correlation_threshold)
        feature.drop(cor_drop, inplace=True, axis=1)

        # woe转化
        print("单变量预处理: WOE转换")
        feature = binning.woe_transform(feature, woe_result, bin_result)

        feature_selected = feature.columns

        # 中间结果保存
        result = {"missing_value": missing_value,
                  "distinct_value": distinct_value,
                  "unique_value": unique_value,
                  "abnormal_value": abnormal_value,
                  "missing_filling": missing_filling,
                  # 'standard_lambda': standard_lambda,
                  'scale_result': scale_result,
                  "bin_result": bin_result,
                  "iv_result": iv_result,
                  "woe_result": woe_result,
                  "woe_adjust_result": woe_adjust_result,
                  "bin_error_drop": bin_error_drop,
                  "iv_drop": iv_drop,
                  "cor_drop": cor_drop,
                  "feature_selected": feature_selected
                  }
        print(">>> 特种工程简单报告:")
        print(f"    缺失值过滤列数量:{missing_value['drop_number']}")
        print(f"    唯一值过滤列数量:{distinct_value['drop_number']}")
        print(f"    众数值过滤列数量:{unique_value['drop_number']}")
        print(f"    IV值过滤列数量:{len(iv_drop)}")
        print(f"    相关性过滤列数量:{len(cor_drop)}")
        print(f"    保留数量:{len(feature_selected)}")
        with open('result/feature_engineering.pickle', 'wb') as f:
            f.write(pickle.dumps(result))

        with open('result/apply_sql.txt', 'w') as f:
            feature_selected = set([v.split("__")[0] for v in result["feature_selected"]])
            sql = ",".join(feature_selected)
            f.write(f"select {sql} from ")
