import pickle
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from mksc.model import model_mapper, model_choose

def training(feature, label, model_name=None):

    # 数据集划分
    x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.2, random_state=0)

    # 模型训练
    result = {}
    if model_name:
        model = model_choose(model_name)
        scores = cross_validate(model, x_train, y_train, cv=5, scoring='precision_macro')
        result[model_name] = scores['test_score'].mean()
        model.fit(x_train, y_train)
    else:
        for m in model_mapper:
            model = model_choose(m)
            scores = cross_validate(model, x_train, y_train, cv=5, scoring='precision_macro')
            result[m] = scores['test_score'].mean()
        model_name = max(result, key=result.get)
        model = model_choose(model_name)
        model.fit(x_train, y_train)

    # 模型评估
    predict_train = model.predict(x_train)
    predict_test = model.predict(x_test)

    acu_train = accuracy_score(y_train, predict_train)
    acu_test = accuracy_score(y_test, predict_test)

    sen_train = recall_score(y_train, predict_train, pos_label=1)
    sen_test = recall_score(y_test, predict_test, pos_label=1)

    spe_train = recall_score(y_train, predict_train, pos_label=0)
    spe_test = recall_score(y_test, predict_test, pos_label=0)
    print(f'模型准确率：训练 {acu_train * 100:.2f}%	测试 {acu_test * 100:.2f}%')
    print(f'正例覆盖率：训练 {sen_train * 100:.2f}%	测试 {sen_test * 100:.2f}%')
    print(f'负例覆盖率：训练 {spe_train * 100:.2f}%	测试 {spe_test * 100:.2f}%')

    # 模型保存
    with open(f'result/{model_name}.pickle', 'wb') as f:
        f.write(pickle.dumps(model))
