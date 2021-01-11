from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier

model_mapper = {"lr": LogisticRegression,
                "dt": DecisionTreeClassifier,
                # "svm": SVC,
                "rf": RandomForestClassifier,
                "gbdt": GradientBoostingClassifier,
                "xgb": XGBClassifier,
                "lgb": LGBMClassifier}

def model_choose(model_name, **kwargs):
    """
    选择模型类
    Args:
        model_name: 模型名称
    Returns:
        对应模型类
    """
    global model_mapper
    return model_mapper[model_name](**kwargs)
