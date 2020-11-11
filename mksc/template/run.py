import eda
import train
import score
import apply


def main():
    tag = input("step 1: 请完成项目配置configuration.ini，完成请键入【Y/y】")
    if tag.upper() != "Y":
        return SystemExit("请完成项目配置")

    tag = input("step2: EDA过程跳过请输入【Y/y】")
    if tag.upper() != "Y":
        report = input("本次运行不生成报告，回车确认取消报告，任意键+回车继续")
        eda.main(report)

    tag = input("step 3: 请完成特征配置variable_type.csv以及自定义配置custom.py，完成请输入【Y/y】，跳过训练过程请输入【S/s】")
    if tag.upper() == "Y":
        print("训练过程...")
        train.main()
    elif tag.upper() != "S":
        return SystemExit("错误参数，退出程序")

    tag = input("step4: 打分过程，跳过请输入【Y/y】")
    if tag.upper() != "Y":
        score.main()

    tag = input("step6: 应用过程，跳过请输入【Y/Y】")
    if tag.upper() != "Y":
        apply.main()


if __name__ == "__main__":
    main()
