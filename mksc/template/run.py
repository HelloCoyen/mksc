import eda
import train
import score
import apply


def main():
    tag = input("step 1: 请完成项目配置configuration.ini，完成请输入【Y】")
    if tag.upper() != "Y":
        return

    tag = input("step2: EDA过程...跳过请输入【S】")
    if tag.upper() != "S":
        eda.main()

    tag = input("step 3: 请完成特征配置variable_type.csv以及自定义配置custom.py，完成请输入【Y】，跳过训练过程请输入【S】")
    if tag.upper() == "Y":
        print("训练过程...")
        train.main()
    elif tag.upper() != "S":
        return

    tag = input("step4: 打分过程....跳过请输入【S】")
    if tag.upper() == "Y" or tag == "S":
        score.main()

    print("step6: 应用过程...")
    apply.main()


if __name__ == "__main__":
    main()
