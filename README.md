# Make Scorecard(mksc)
快速构建二分类模型,标准化特征工程以及拓展制作评分卡,文件说明见`doc\说明手册.md`

## 1. 安装工具包
```
pip install mksc
```

## 2. 创建项目
命令行工具创建项目
```
mksc project_name
```

## 3. 修改项目配置
修改`project_name\config\configuration.ini`文件，进行项目配置


## 4. 探索性数据分析
进行探索性数据分析`python project_name\eda.py`  
生成：  
* 数据报告： `project_name\result\report.html`  
* 抽样数据： `project_name\result\sample.xlsx`
* 特征配置： `project_name\config\variable_type.csv`

## 5. 修改特征配置
修改`project_name\config\variable_type.csv`文件，进行特征配置，配置列说明如下：  
* __isSave__：变量是否保留进行特征工程
    - 取值：0-不保留；1-保留
* __Type__: 变量类型
    - 取值： numeric-数值类型；category-类别类型；datetime-日期类型；label-标签列

## 5. 自定义数据清洗
编写自定义数据清洗与特征组合过程函数`project_name\custom.py`。  
自定义过程封装在Custom类中，定义了3个静态方法，`clean_data`用于处理行方向的数据与值修改，
`feature_combination`用于扩展列，`model`用于替换训练模型。

## 6. 训练模块
完成以上配置后，执行训练`python project_name\train.py`。  
模型结果、特征工程结果均置于`project_name\result`下
至此完成二分类项目构建

## 7. 评分卡与模型调整
`python project_name\score.py`  
TODO `python project_name\adjust.py`

## 8. 模型应用与预测
TODO `python project_name\apply.py`  
TODO `PYTHON project_name\run_project_name.py`  