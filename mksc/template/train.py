from statsmodels.iolib.smpickle import load_pickle
import mksc
from mksc.model import training
from mksc.feature_engineering import preprocess
from mksc.feature_engineering import transform
from custom import Custom
import argparse

def main(**kwargs):

    feature_engineering = load_pickle('result/feature_engineering.pickle')
    data = mksc.load_data()
    numeric_var, category_var, datetime_var, label_var = preprocess.get_variable_type()
    feature = data[numeric_var + category_var + datetime_var]
    label = data[label_var]

    # 自定义数据清洗
    feature, label = Custom.clean_data(feature, label)

    # 数据类型转换
    feature[numeric_var] = feature[numeric_var].astype('float')
    feature[category_var] = feature[category_var].astype('object')
    feature[datetime_var] = feature[datetime_var].astype('datetime64')

    # 自定义特征组合模块
    feature = Custom.feature_combination(feature)
    
    feature = feature[feature_engineering['feature_selected']]

    # 数据处理
    feature = transform(feature, feature_engineering)

    # 模型训练
    training(feature, label, **kwargs)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-m", "--model", type=str, help="模型选择，默认自动选择最优")
    args.add_argument("-r", "--resample", type=bool, help="是否使用重采样，默认不采样")
    accepted = vars(args.parse_args())
    main(model_name=accepted['model'], resample=accepted['resample'])
