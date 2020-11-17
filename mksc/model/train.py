import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, auc, roc_curve
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from mksc.model import model_mapper, model_choose


def training(feature, label, model_name=None, **kwargs):

    print(f"样本比例：label.sum()/len(label) ")
    # 重采样
    if "resample_threshold" in kwargs.keys():
        if (label.sum()/len(label) < kwargs["resample_threshold"]
                or label.sum()/len(label) > 1 - kwargs["resample_threshold"]):
            feature, label = SMOTE().fit_sample(feature, label)
    # 数据集划分
    x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.2, random_state=0)

    # 模型训练
    result = {}
    if model_name:
        model = model_choose(model_name, **kwargs)
        scores = cross_validate(model, x_train, y_train, cv=5, scoring='roc_auc')
        result[model_name] = scores['test_score'].mean()
        print(f"Model-{model_name}: roc_auc: {result[model_name]:.2f}")
        model.fit(x_train, y_train)
    else:
        for m in model_mapper:
            model = model_choose(m)
            scores = cross_validate(model, x_train, y_train, cv=5, scoring='roc_auc')
            result[m] = scores['test_score'].mean()
            print(f"Model-{m}: roc_auc: {result[m]:.2f}")
        model_name = max(result, key=result.get)
        model = model_choose(model_name)
        model.fit(x_train, y_train)

    # ROC曲线&阈值计算
    predict_train = np.array([i[1] for i in model.predict_proba(x_train)])
    fpr, tpr, thresholds = roc_curve(y_train.values, predict_train, pos_label=1)
    auc_score = auc(fpr, tpr)
    w = tpr - fpr
    ks_score = w.max()
    ks_x = fpr[w.argmax()]
    ks_y = tpr[w.argmax()]
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label='AUC=%.5f' % auc_score)
    ax.set_title('Receiver Operating Characteristic')
    ax.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))
    ax.plot([ks_x, ks_x], [ks_x, ks_y], '--', color='red')
    ax.text(ks_x, (ks_x + ks_y) / 2, '  KS=%.5f' % ks_score)
    ax.legend()
    fig.savefig(f'result/{model_name}_roc.png')

    # 模型评估
    predict_train = [0 if i[1] < thresholds[w.argmax()] else 1 for i in model.predict_proba(x_train)]
    predict_test = [0 if i[1] < thresholds[w.argmax()] else 1 for i in model.predict_proba(x_test)]

    acu_train = accuracy_score(y_train, predict_train)
    acu_test = accuracy_score(y_test, predict_test)

    sen_train = recall_score(y_train, predict_train, pos_label=1)
    sen_test = recall_score(y_test, predict_test, pos_label=1)

    spe_train = recall_score(y_train, predict_train, pos_label=0)
    spe_test = recall_score(y_test, predict_test, pos_label=0)
    print(f"K-S 值： {ks_score}")
    print(f'模型准确率：训练 {acu_train * 100:.2f}%	测试 {acu_test * 100:.2f}%')
    print(f'正例覆盖率：训练 {sen_train * 100:.2f}%	测试 {sen_test * 100:.2f}%')
    print(f'负例覆盖率：训练 {spe_train * 100:.2f}%	测试 {spe_test * 100:.2f}%')

    # 模型保存
    result = (model, thresholds[w.argmax()])
    with open(f'result/{model_name}.pickle', 'wb') as f:
        f.write(pickle.dumps(result))
