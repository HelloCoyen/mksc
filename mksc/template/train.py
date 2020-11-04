import mksc
from mksc.feature_engineering import preprocess
from mksc.feature_engineering import FeatureEngineering
from mksc.model import training
from custom import Custom

def main(model_name=None):
    """
    项目训练程序入口
    """
    # 加载数据、变量类型划分、特征集与标签列划分
    data = mksc.load_data("pickle")
    numeric_var, category_var, datetime_var, label_var = preprocess.get_variable_type()
    feature = data[numeric_var + category_var + datetime_var]
    label = data[label_var]

    # 自定义数据清洗
    feature, label = Custom.clean_data(feature, label)

    # 数据类型转换
    feature[numeric_var] = feature[numeric_var].astype('float')
    feature[category_var] = feature[category_var].astype('object')
    feature[datetime_var] = feature[datetime_var].astype('datetime64')

    # 自定义特征组合
    feature = Custom.feature_combination(feature)

    # 标准化特征工程
    fe = FeatureEngineering(feature, label)
    feature = fe.run()

    # 模型训练
    training(feature, label, model_name)


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        main()

