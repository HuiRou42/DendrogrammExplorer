
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

iris = load_iris()
data = iris.data  # 数据
lables = iris.target  # 标签
feaures = iris.feature_names
df = pd.DataFrame(data, columns=feaures)  # 原始数据


# 第一步：无量纲化
def standareData(df):
    """
    df : 原始数据
    return : data 标准化的数据
    """
    data = pd.DataFrame(index=df.index)  # 列名，一个新的dataframe
    columns = df.columns.tolist()  # 将列名提取出来
    for col in columns:
        d = df[col]
        max = d.max()
        min = d.min()
        mean = d.mean()
        data[col] = ((d - mean) / (max - min)).tolist()
    return data


#  某一列当做参照序列，其他为对比序列
def graOne(Data, m=0):
    """
    return:
    """
    columns = Data.columns.tolist()  # 将列名提取出来
    # 第一步：无量纲化
    data = standareData(Data)
    referenceSeq = data.iloc[:, m]  # 参考序列
    data.drop(columns[m], axis=1, inplace=True)  # 删除参考列
    compareSeq = data.iloc[:, 0:]  # 对比序列
    row, col = compareSeq.shape
    # 第二步：参考序列 - 对比序列
    data_sub = np.zeros([row, col])
    for i in range(col):
        for j in range(row):
            data_sub[j, i] = abs(referenceSeq[j] - compareSeq.iloc[j, i])
    # 找出最大值和最小值
    maxVal = np.max(data_sub)
    minVal = np.min(data_sub)
    cisi = np.zeros([row, col])
    for i in range(row):
        for j in range(col):
            cisi[i, j] = (minVal + 0.5 * maxVal) / (data_sub[i, j] + 0.5 * maxVal)
    # 第三步：计算关联度
    result = [np.mean(cisi[:, i]) for i in range(col)]
    result.insert(m, 1)  # 参照列为1
    return pd.DataFrame(result)


def GRA(Data):
    df = Data.copy()
    columns = [str(s) for s in df.columns if s not in [None]]  # [1 2 ,,,12]
    # print(columns)
    df_local = pd.DataFrame(columns=columns)
    df.columns = columns
    for i in range(len(df.columns)):  # 每一列都做参照序列，求关联系数
        df_local.iloc[:, i] = graOne(df, m=i)[0]
    df_local.index = columns
    return df_local


# Show heatmap
def ShowGRAHeatMap(DataFrame):
    colormap = plt.cm.hsv
    ylabels = DataFrame.columns.values.tolist()
    f, ax = plt.subplots(figsize=(15, 15))
    ax.set_title('Wine GRA')
    # 设置展示一半，如果不需要注释掉mask即可
    mask = np.zeros_like(DataFrame)
    mask[np.triu_indices_from(mask)] = True  # np.triu_indices 上三角矩阵

    with sns.axes_style("white"):
        sns.heatmap(DataFrame,
                    cmap="YlGnBu",
                    annot=True,
                    mask=mask,
                    )
    plt.show()


data_wine_gra = GRA(df)
ShowGRAHeatMap(data_wine_gra)