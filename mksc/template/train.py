import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve, accuracy_score, recall_score
import mksc
from mksc.feature_engineering import preprocess
from mksc.feature_engineering import FeatureEngineering
from custom import Custom

def training(x_train, y_train, x_test, y_test, x_valid, y_valid):
    """
    模型训练过程函数
    1. 训练
    2. 预测
    3. 评估
    4. TODO 模型可选

    Args:
        x_train: 训练集特征
        y_train: 训练集标签
        x_test: 测试集特征
        y_test: 测试集标签
        x_valid: 验证集特征
        y_valid: 验证集标签
    """
    model = LogisticRegression()
    model.fit(x_train, y_train)

    # 预测结果
    predict_train = model.predict(x_train)
    predict_valid = model.predict(x_valid)
    predict_test = model.predict(x_test)

    # 模型评估
    acu_train = accuracy_score(y_train, predict_train)
    acu_valid = accuracy_score(y_valid, predict_valid)
    acu_test = accuracy_score(y_test, predict_test)

    sen_train = recall_score(y_train, predict_train, pos_label=1)
    sen_valid = recall_score(y_valid, predict_valid, pos_label=1)
    sen_test = recall_score(y_test, predict_test, pos_label=1)

    spe_train = recall_score(y_train, predict_train, pos_label=0)
    spe_valid = recall_score(y_valid, predict_valid, pos_label=0)
    spe_test = recall_score(y_test, predict_test, pos_label=0)
    print(f'模型准确率：验证 {acu_valid * 100:.2f}%	训练 {acu_train * 100:.2f}%	测试 {acu_test * 100:.2f}%')
    print(f'正例覆盖率：验证 {sen_valid * 100:.2f}%	训练 {sen_train * 100:.2f}%	测试 {sen_test * 100:.2f}%')
    print(f'负例覆盖率：验证 {spe_valid * 100:.2f}%	训练 {spe_train * 100:.2f}%	测试 {spe_test * 100:.2f}%')

    # K-s & roc
    predict_train_prob = np.array([i[1] for i in model.predict_proba(x_train)])
    fpr, tpr, thresholds = roc_curve(y_train.values, predict_train_prob, pos_label=1)
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
    fig.savefig("result/ks_roc.png")

    # 模型保存
    coefs = dict(list(zip(x_train.columns, list(model.coef_[0]))) + [("intercept_", model.intercept_[0])])
    with open('result/model.pickle', 'wb') as f:
        f.write(pickle.dumps(model))
    with open('result/coefs.pickle', 'wb') as f:
        f.write(pickle.dumps(coefs))


def main():
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

    # 数据集划分
    x_train, x_valid_test, y_train, y_valid_test = train_test_split(feature, label, test_size=0.4, random_state=0)
    x_valid, x_test, y_valid, y_test = train_test_split(x_valid_test, y_valid_test, test_size=0.5, random_state=0)

    # 模型训练
    training(x_train, y_train, x_test, y_test, x_valid, y_valid)


if __name__ == "__main__":
    main()
