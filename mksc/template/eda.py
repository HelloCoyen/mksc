import pandas as pd
import pandas_profiling as pp
from statsmodels.iolib.smpickle import save_pickle
import argparse
import mksc

def main(report=False, apply=False, local=False):
    """
    探索性数据分析主程序入口
    Args:
        report: 是否保留分析报告，由于该报告对大数据集会占用大量时间，默认不生成
        apply: 是否读取应用集并保存pickle文件，此处注意应用集规模，默认生成
    生成以下四份文件：
        1、数据对象文件：         “data/data.pickle”
        2、变量类型配置文件：      “config/variable_type.csv”
        3、样例数据：             “data/sample.xlsx”
        4、结果报告：             “result/report.html”
    """
    # 加载数据并保存本地
    data = mksc.load_data(mode="all", local=local)
    save_pickle(data, 'data/data.pickle')

    # 生成变量类别配置文件
    # 变量是否保留进行特征工程(isSave): 0-不保留；1-保留
    # 变量类型(Type): numeric-数值类型；category-类别类型；datetime-日期类型；label-标签列；identifier-业务标识符
    res = pd.DataFrame(zip(data.columns,
                           [1]*len(data.columns),
                           ['numeric']*len(data.columns),
                           ['']*len(data.columns)),
                       columns=['Variable', 'isSave:[0/1]', 'Type:[identifier/numeric/category/datetime/label]', 'Comment'])
    res.to_csv("config/variable_type.csv", index=False)

    # 抽样探索
    sample = data.sample(min(len(data), 1000))
    sample.reset_index(drop=True, inplace=True)
    with pd.ExcelWriter('data/sample.xlsx') as writer:
        sample.to_excel(writer, sheet_name="抽样数据1000条", index=False)
        sample.corr().to_excel(writer, sheet_name="相关系数", index=False)
        (sample
            .select_dtypes(exclude=['object', 'datetime'])
            .describe()
            .to_excel(writer, sheet_name="数值数据汇总"))
        (sample
            .select_dtypes(include=['object', 'datetime'])
            .describe()
            .to_excel(writer, sheet_name="分类数据汇总"))

    # 保存分析报告
    if report and (not sample.empty):
        report = pp.ProfileReport(sample)
        report.to_file('result/report.html')

    # 保存应用集，注意规模
    if apply:
        apply = mksc.load_data(mode="apply", local=False)
        save_pickle(apply, 'data/apply.pickle')


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("-r", "--report", type=bool, default=False, help="是否生成报告,默认不生成")
    args.add_argument("-a", "--apply", type=bool, default=False, help="是否保存应用集二进制文件,默认不生成")
    args.add_argument("-l", "--local", type=bool, default=False, help="是否读取本地文件,默认不读取")
    accepted = vars(args.parse_args())
    main(report=accepted.get("report"), apply=accepted.get("apply"), local=accepted.get("local"))
