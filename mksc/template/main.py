import eda
import train
import score
import apply

def main():
    tag = input("step 1: 请完成项目配置，完成请输入【Y】")
    if tag == "Y":
        print("step2: EDA过程...")
        eda.main()
        tag2 = input("step 3: 请完成特征配置，完成请输入【Y】")
        if tag2 == "Y":
            print("step4: 训练过程...")
            train.main()
            print("step5: 打分过程...")
            score.main()
            print("step6: 应用过程...")
            apply.main()

if __name__ == "__main__":
    main()
