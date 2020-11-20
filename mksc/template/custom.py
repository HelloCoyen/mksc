import pandas as pd


class Custom(object):
    """
    自定义预处理类函数封装
    """
    def __init__(self):
        pass

    @staticmethod
    def clean_data(feature, label):
        """
        基于单元格与行操作的数据清洗

        Args:
            feature: 待清洗特征数据框
            label: 待清洗标签序列

        Returns:
            feature_tmp: 已清洗特征数据框
            label_tmp: 已清洗标签序列
        """
        feature_tmp = feature.copy()
        label_tmp = label.copy()
        # 默认值处理
        # 单位处理
        # 正则处理
        # 业务缺失值补充
        # 业务异常值剔除
        return feature_tmp, label_tmp

    @staticmethod
    def feature_combination(feature):
        """
        基于列操作的数据清洗与特征构造

        Args:
            feature: 待清洗特征数据框

        Returns:
            feature_tmp: 已清洗特征数据框
        """
        feature_tmp = feature.copy()

        # 构造衍生变量, 提取完后需要丢弃变量
        # eg: feature_tmp['new_variable'] = feature_tmp['old_variable']
        # eg: feature_tmp.drop('old_variable', axis=1, inplace=True)

        # 日期变量处理, 提取完后需要丢弃变量
        datetime_var = feature_tmp.select_dtypes(include='datetime64').columns
        date_var = ['day', 'dayofweek', 'dayofyear', 'days_in_month', 'is_leap_year', 'is_month_end',
                    'is_month_start', 'is_quarter_end', 'is_quarter_start', 'is_year_end',
                    'is_year_start', 'month', 'quarter', 'week', 'weekday', 'weekofyear', 'year']
        for dv in datetime_var:
            try:
                """字段组合提取操作，避免特征表过大，应用时使用使用不同特征表,使用try-except语句"""
                for v in date_var:
                    feature_tmp[f"{dv}__{v}"] = eval(f"feature_tmp[{dv}].dt.{v}")
                    feature_tmp[f"{dv}__{v}"] = feature_tmp[f"{dv}__{v}"].astype("object")
            except KeyError:
                pass
        feature_tmp.drop(datetime_var, axis=1, inplace=True)
        return feature_tmp

    @staticmethod
    def feature_adjust(feature, feature_srouce):
        adjust_var = []
        feature_tmp = feature[adjust_var]
        """
        adjust_var单独特征处理
        """
        new_feature = pd.concat([feature_tmp, feature_srouce[adjust_var]], axis=1)
        return new_feature

class CustomTrainModel(object):
    """
    TODO自定义模型, 必须包括以下三个方法
    """
    def fit(self, x, y):
        pass

    def predict(self, x):
        return

    def predict_proba(self, x):
        return

class CustomApply(object):
    """
    TODO自定义应用集处理
    """
    def load_data(self):
        pass

    def load_model(self):
        pass

    def predict(self):
        pass


if __name__ == "__main__":

    import mksc
    from mksc.feature_engineering import preprocess
    # 加载数据、变量类型划分、特征集与标签列划分
    data = mksc.load_data()
    numeric_var, category_var, datetime_var, label_var = preprocess.get_variable_type()
    feature = data[numeric_var + category_var + datetime_var]
    label = data[label_var]
    feature[numeric_var] = feature[numeric_var].astype('float')
    feature[category_var] = feature[category_var].astype('object')
    feature[datetime_var] = feature[datetime_var].astype('datetime64')
