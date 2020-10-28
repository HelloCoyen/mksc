import pandas_profiling as pp
import pandas as pd
from statsmodels.iolib.smpickle import save_pickle
import mksc

def main():
    """
    探索性数据分析主程序入口
    生成结果报告、配置文件与样例数据
    """
    # 加载数据并保存本地
    data = mksc.load_data()
    save_pickle(data, 'data/data.pickle')

    # 保存分析报告
    sample = data.sample(min(len(data), 1000))
    sample.reset_index(drop=True, inplace=True)
    report = pp.ProfileReport(sample)
    report.to_file('result/eda_report.html')

    # 保存简单样本
    sample.to_excel('data/data_sample.xlsx', index=False)

    # 生成变量类别配置文件
    # 变量是否保留进行特征工程(isSave): 0-不保留；1-保留
    # 变量类型(Type): numeric-数值类型；category-类别类型；datetime-日期类型；label-标签列
    res = pd.DataFrame(zip(data.columns, [1]*len(data.columns), ['category']*len(data.columns), ['']*len(data.columns)),
                       columns=['Variable', 'isSave:[0/1]', 'Type:[numeric/category/datetime/label]', 'Comment'])
    res.to_csv("config/variable_type.csv", index=False)


if __name__ == '__main__':
    main()
