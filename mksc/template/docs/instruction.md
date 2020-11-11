# 处理过程
1. 探索性数据分析
2. 特征分类
3. 自定义处理
4. 特征工程
5. 模型训练
6. 评分卡制作
7. 预测评分

# 生成中间文件
1. config目录
    * configuration.ini: 初始文件，项目配置信息
    * variable_type.csv: 探索性数据分析脚本(eda.py)生成的特征配置文件
2. data目录
    * data.pickle: 探索性数据分析脚本(eda.py)生成的数据对象文件有标签
    * apply.pickle: 探索性数据分析脚本(eda.py)生成的数据对象文件无标签
    * sample.xlsx: 探索性数据分析脚本(eda.py)生成的数据抽样文件
    * train.csv\test.csv\gender_submission.csv: 泰坦尼克数据集，demo用
3. docs目录
    * instruction.md： 教程
    * baseline.md： 模型思路
4. result目录
    * apply_result.csv: 应用脚本(apply.py)生成的结果数据
    * card.pickle: 针对逻辑回归评分卡制作(score.py)生成评分卡结果对象文件
    * card.xlsx: 针对逻辑回归评分卡制作(score.py)生成评分卡结果文件
    * coefs.pickle: 针对逻辑回归训练脚本(train.py)生成的特征对象文件
    * feature_engineering.pickle: 训练脚本(train.py)生成的特征工程结果对象文件
    * {model_name}_roc.png: 训练脚本(train.py)生成的模型评估图
    * {model_name}.pickle: 训练脚本(train.py)生成的模型结果文件
    * report.html: 探索性数据分析脚本(eda.py)生成的可视化报告
    
# 包使用流程
1. 配置配置项configuration.ini  
    主要配置数据读取路径,如果同时配置本地与远程,只有远程生效
2. EDA探索过程  
    直接运行eda.py,生成探索性数据分析报告report.html,结合改报告,配置生成配置文件variable_type.csv限定变量类型  
3. 自定义特征清洗与组合  
    编辑custom.py,修改数据清洗与特征提取方法。
4. 模型训练  
    直接运行train.py,开始训练,保存模型文件model.pickle
5. 生成评分卡  
    直接运行score.py,开始制作评分卡，保存评分卡文件card.xlsx
6. 模型应用   
    直接运行apply.py,进行模型应用
