from statsmodels.iolib.smpickle import load_pickle
import pandas as pd
import mksc
from mksc.feature_engineering import preprocess
from mksc.feature_engineering import scoring
from mksc.feature_engineering import transform
from custom import Custom

def main():

    # 数据、模型加载
    model = load_pickle('result/model.pickle')
    coefs = load_pickle('result/coefs.pickle')
    feature_engineering = load_pickle('result/feature_engineering.pickle')

    data = mksc.load_data()
    numeric_var, category_var, datetime_var, label_var = preprocess.get_variable_type()
    feature = data[numeric_var + category_var + datetime_var]
    label = []

    # 自定义数据清洗
    feature, label = Custom.clean_data(feature, label)

    # 数据类型转换
    feature[numeric_var] = feature[numeric_var].astype('float')
    feature[category_var] = feature[category_var].astype('object')
    feature[datetime_var] = feature[datetime_var].astype('datetime64')
    feature = data[list(coefs.keys())[:-1]]

    # 自定义特征组合模块
    feature = Custom.feature_combination(feature)
    # 数据处理
    feature = transform(feature, feature_engineering)
    res = pd.DataFrame(model.predict(feature), columns=['label'])
    res = pd.concat([feature, res], axis=1)

    # 转化评分
    score_card = load_pickle('result/card.pickle')
    res = scoring.transform_score(res, score_card)
    # 结果保存
    mksc.save_result(res)


if __name__ == '__main__':
    main()
