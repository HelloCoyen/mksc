import mksc
from mksc.feature_engineering import preprocess
from mksc.feature_engineering import FeatureEngineering
from custom import Custom
import argparse

def main(**kwargs):
    """
    项目特征工程程序入口
    """
    # 加载数据、变量类型划分、特征集与标签列划分
    data = mksc.load_data()
    numeric_var, category_var, datetime_var, label_var, identifier_var = preprocess.get_variable_type()
    feature = data[numeric_var + category_var + datetime_var]
    label = data[label_var]

    # 自定义数据清洗
    feature, label = Custom.clean_data(feature, label)

    # 数据类型转换
    feature[numeric_var] = feature[numeric_var].astype('float')
    feature[category_var] = feature[category_var].astype('object')
    feature[datetime_var] = feature[datetime_var].astype('datetime64')

    # 自定义特征组合，全部为数值变量
    feature = Custom.feature_combination(feature)

    # 标准化特征工程
    fe = FeatureEngineering(feature, label, **kwargs)
    fe.run()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-m", "--missing_threshold", type=tuple, default=(0.95, 0.05), help="缺失值阈值,默认(0.95, 0.05)")
    args.add_argument("-d", "--distinct_threshold", type=float, default=0.95, help="唯一率阈值,默认0.95")
    args.add_argument("-u", "--unique_threshold", type=float, default=0.95, help="众数阈值,默认0.95")
    args.add_argument("-a", "--abnormal_threshold", type=float, default=0.05, help="极端值阈值,默认0.05")
    args.add_argument("-c", "--correlation_threshold", type=float, default=0.8, help="相关系数阈值,默认0.8")
    args.add_argument("-v", "--variance_threshold", type=float, default=0.05, help="方差阈值,默认0.05")
    accepted = vars(args.parse_args())
    missing_threshold = accepted.get("missing_threshold")
    distinct_threshold = accepted.get("distinct_threshold")
    unique_threshold = accepted.get("unique_threshold")
    abnormal_threshold = accepted.get("abnormal_threshold")
    correlation_threshold = accepted.get("correlation_threshold")
    variance_threshold = accepted.get("variance_threshold")
    main(missing_threshold=missing_threshold, 
         distinct_threshold=distinct_threshold, 
         unique_threshold=unique_threshold,
         abnormal_threshold=abnormal_threshold, 
         correlation_threshold=correlation_threshold, 
         variance_threshold=variance_threshold)
