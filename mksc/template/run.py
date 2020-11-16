import eda
import train
import score
import apply
import feature

def main():
    tag = input(">>> step 1: 请完成项目配置configuration.ini，完成后请键入【Y/y】继续，任意键退出")
    if tag.upper() != "Y":
        return SystemExit(">>> 请完成项目配置")

    tag = input(">>> step 2: EDA过程跳过请输入【Y/y】")
    if tag.upper() != "Y":
        report = input(">>> 报告生成将花费大量时间，回车确认取消报告，任意键回车继续生成")
        eda.main(report)

    tag = input(">>> step 3: 请完成特征配置variable_type.csv以及自定义配置custom.py，完成请输入【Y/y】，跳过特征过程请输入【S/s】")
    if tag.upper() == "Y":
        print(">>> 特征过程...")
        feature.main()
    elif tag.upper() != "S":
        return SystemExit(">>> 错误参数，退出程序")

    tag = input(">>> step 4: 训练过程，跳过过程请输入【S/s】")
    if tag.upper() != "S":
        print(">>> 训练过程...")
        train.main()
    elif tag.upper() != "S":
        return SystemExit(">>> 错误参数，退出程序")

    tag = input(">>> step4: 打分过程，跳过请输入【Y/y】")
    if tag.upper() != "Y":
        score.main()

    tag = input(">>> step6: 应用过程，跳过请输入【Y/Y】")
    if tag.upper() != "Y":
        apply.main()


if __name__ == "__main__":
    main()
