import eda
import train
import score
import apply


def main():
    tag = input("step 1: 请完成项目配置，完成请输入【Y】")
    if tag == "Y":
        pass
    else:
        return

    tag = input("step2: EDA过程...跳过请输入【S】")
    if tag == "S":
        pass
    else:
        eda.main()

    tag = input("step 3: 请完成特征配置以及自定义配置，完成请输入【Y】， 跳过训练过程请输入【S】")
    if tag == "S":
        pass
    elif tag == "Y":
        print("step4: 训练过程...")
        train.main()
    else:
        return

    tag = input("step5: 打分过程....跳过请输入【S】")
    if tag == "Y" or tag == "S":
        score.main()
        print("step6: 应用过程...")
        apply.main()


if __name__ == "__main__":
    main()
