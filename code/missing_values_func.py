import pandas as pd
import numpy as np


def missing_values_table(data_frame):
    # 计算总的缺失值
    miss_value = data_frame.isnull().sum()

    # 计算缺失值的百分比
    miss_value_percent = 100 * data_frame.isnull().sum() / len(data_frame)

    # 把结果制成表格
    miss_value_table = pd.concat([miss_value, miss_value_percent], axis=1)

    # 对列重命名，第一列：Missing Values，第二列：% of Total Values
    miss_value_table_rename_columns = miss_value_table.rename(columns={0: 'Missing Values', 1: '% of Total Values'})

    # 根据百分比对表格进行降序排列
    miss_value_table_rename_columns = miss_value_table_rename_columns[
        miss_value_table_rename_columns.iloc[:, 1] != 0].sort_values('% of Total Values', ascending=False).round(1)

    # 打印总结信息：总的列数，有数据缺失的列数
    print("Your selected dataframe has " + str(data_frame.shape[1]) + " columns.\n"
          "There are " + str(miss_value_table_rename_columns.shape[0]) + " columns that have missing values.")

    # 返回带有缺失值信息的 dataframe
    return miss_value_table_rename_columns


# def remove_collinear_features(x, threshold):
#     """
#     Objective:
#        删除数据帧中相关系数大于阈值的共线特征。删除共线特征可以帮助模型泛化并提高模型的可解释性。
#
#     Inputs:
#         阈值：删除任何相关性大于此值的特征
#
#     Output:
#         仅包含非高共线特征的数据帧
#     """
#
#     # 不要删除能源之星得分之间的相关性
#     y = x['score']  # 在原始数据 x 中， score 当作 y 值
#     x = x.drop(columns=['score'])  # 去除标签值以外的当作特征
#
#     while True:
#         # 计算一个矩阵，两两的相关系数
#         corr_matrix = x.corr()
#
#         for i in range(len(corr_matrix)):
#             corr_matrix.iloc[i][i] = 0  # 将对角线上的相关系数置为0，避免自己跟自己计算相关系数大于阈值
#
#         # 定义待删除的特征
#         drop_cols = []
#
#         for col in corr_matrix:
#             if col not in drop_cols:  # A和B，B和A的相关系数一样，避免AB全删了
#                 # 取出每一列的相关系数，取的是相关系数的绝对值
#                 v = np.abs(corr_matrix[col])
#                 # 如果相关系数大于设置的阈值
#                 if np.max(v) > threshold:
#                     # 取出最大值对应的索引
#                     index = np.argmax(v)  # 找到最大值的列名
#                     name = x.columns[index]
#                     drop_cols.append(name)
#
#         # 列表不为空，就删除，列表为空，符合条件，推出循环
#         if drop_cols:
#             x = x.drop(columns=drop_cols, axis=1)
#         else:
#             break
#
#     # 指定标签
#     x['score'] = y
#
#     return x


def remove_collinear_features(x, threshold):
    """
    Objective:
       删除数据帧中相关系数大于阈值的共线特征。 删除共线特征可以帮助模型泛化并提高模型的可解释性。

    Inputs:
        阈值：删除任何相关性大于此值的特征

    Output:
        仅包含非高共线特征的数据帧
    """

    # 不要删除能源之星得分之间的相关性
    y = x['score']
    x = x.drop(columns=['score'])

    # 计算相关性矩阵
    corr_matrix = x.corr()
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []

    # 迭代相关性矩阵并比较相关性
    for i in iters:
        for j in range(i):
            item = corr_matrix.iloc[j:(j+1), (i+0):(i+1)]
            col = item.columns
            row = item.index
            val = abs(item.values)

            if val >= threshold:
                # 打印有相关性的特征和相关值
                # print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                drop_cols.append(col.values[0])

    # 删除重复元素
    drops = set(drop_cols)
    x = x.drop(columns=drops)
    # x = x.drop(columns=['Weather Normalized Site EUI (kBtu/ft²)',
    #                       'Water Use (All Water Sources) (kgal)',
    #                       'log_Water Use (All Water Sources) (kgal)',
    #                       'Largest Property Use Type - Gross Floor Area (ft²)'])

    # 将得分添加回数据
    x['score'] = y

    return x



# mae 平均的绝对值，就是 (真实值 - 预测值) / n
# abs(): 绝对值
def mae(y_true, y_pred):
    return np.mean(abs(y_true - y_pred))


# 接受模型，训练模型，并在测试集上评估模型
def fit_and_evaluate(model, X, y, X_test, y_test):
    # 训练模型
    model.fit(X, y)

    # 做出预测和评估
    model_pred = model.predict(X_test)
    model_mae = mae(y_test, model_pred)

    # 返回性能指标
    return model_mae
