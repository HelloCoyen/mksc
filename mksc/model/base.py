from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier
from lightgbm import LGBMClassifier

model_mapper = {"lr": LogisticRegression,
                "dt": DecisionTreeClassifier,
                "svm": SVC,
                "rf": RandomForestClassifier,
                "gbdt": GradientBoostingClassifier,
                "xgb": XGBClassifier,
                "lgb": LGBMClassifier}

def model_choose(model_name, **kwargs):
    global model_mapper
    return model_mapper[model_name](**kwargs)
