数据挖掘基本路线：  
数据获取 -> 数据清洗 -> 模型训练 -> 模型评估 -> 模型调整 -> 应用预测

1. 加载数据与变量类型

2. 特征集划分

3. 此处引入自定义数据预处理模块，以处理不同数据的清洗问题（时间特征处理、业务异常值）

4. 数据类型转换

5. 此处引入自定义数据预处理模块，进行特征组合

6. 分类特征one-hot编码

7. 删除分类特征与时间特征

7. 特征选择Filter:
    基于描述性统计特征筛选，完成一轮筛选
    * 方差选择法：
    * 缺失值过滤：缺失大于90%的变量剔除
    * 唯一标识过滤：类别超过90%的剔除
    * 唯一率值过滤：众数超过90%的剔除

7. 剩余特征分类

8. 极端值处理
    极端值低于5%的用空值填充，其余分箱处理

8.5 TODO异常值处理：太难了不做

9. 缺失值处理
    缺失比例低于5%的用均值和众数填充

10 数据归一化：暂时不需要

11 数据正态化：暂时不需要

10. 相关性统计分析

11. 数值变量最优分箱

11.5 TODO PSI筛选:好累不做

12. 相关性筛选

13. IV筛选

14. woe转化

16. 样本平衡性调整

17. 逐步回归筛选 TODO 层次聚类降维

18. 数据集划分

19. 模型训练

20. 模型评估与调优模型选择

21. 模型保存

22. 模型调整+自定义配置

23. 模型应用