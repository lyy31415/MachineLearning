import time

import pandas as pd
import warnings
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
import seaborn as sns
from sklearn.model_selection import train_test_split
from missing_values_func import missing_values_table, remove_collinear_features, mae, fit_and_evaluate

# 预处理： 缺失值、最大最小归一化
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

# 机器学习算法库
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# 调参工具包
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

# LIME用于解释预测
import lime
import lime.lime_tabular
from sklearn import tree

warnings.filterwarnings("ignore")

# 绘图全局的设置好了，画图字体大小
plt.rcParams['font.size'] = 24

sns.set(font_scale=2)

#  API要升级或遗弃了，不想看就设置一下warning
pd.options.mode.chained_assignment = None

# 经常用到head()，最多展示多少条数
pd.set_option("display.max.columns", 60)

# 1. 数据清理和格式化
# 1.1 加载并检查数据
df = pd.read_csv('../data/Energy_and_Water_Data_Disclosure_for_Local_Law_84_2017__Data_for_Calendar_Year_2016_.csv',
                 ',')
# print(df.head(3))
# print(df.info())

# 1.2 数据类型和缺失值
# 1.2.1 将数据转换为正确的类型
# 将“Not Available”项替换为可以解释为浮点数的np.nan
df = df.replace({'Not Available': np.nan})
# print(df.info())
# print(df.columns)

# 一些明确包含数字（例如ft²）的列被存储为object类型。 我们不能对字符串进行数值分析，
# 因此必须将其转换为数字（特别是浮点数）数据类型
# 对列数进行迭代
for col in list(df.columns):
    # 选择需要被数字化的列，通过if 判断实现
    # 凡是包含下列红色字体的列，都需要被转化为数据类型
    if ('ft²' in col or 'kBtu' in col or 'Metric Tons CO2e' in col or 'kWh' in col or
            'therms' in col or 'gal' in col or 'Score' in col):
        # 将数据类型转换为float
        df[col] = df[col].astype(float)

# print(df.describe())

# 1.3 处理缺失值
# 按列计算缺失值的函数
missing_df = missing_values_table(df)
# print(missing_df)

# 获取缺失值超过50％的列
missing_columns = list(missing_df[missing_df['% of Total Values'] > 50].index)
print('We will remove %d columns.' % len(missing_columns))

# 删除缺失值超过50％的列
df = df.drop(columns=list(missing_columns), axis=1)

# 2. 探索性数据分析（EDA）： 我们的数据可以告诉我们什么，是否存在某些异常、模式、关系等
# 2.1 单变量图: 图表显示单个变量的分布，例如直方图
# figsize(13, 13)
#
# # 将Energy Star Score重新命名为score
df = df.rename(columns={'ENERGY STAR Score': 'score'})
#
# # 绘制直方图
# # 在seaborn中找到不同的风格
# plt.style.use('fivethirtyeight')
#
# # dropna(): 该函数主要用于滤除缺失数据
# plt.hist(df['score'].dropna(), bins=100, edgecolor='k')
# # x,y 坐标轴标签
# plt.xlabel('Score')
# plt.ylabel('Number of Buildings')
#
# # 图表名称
# plt.title('Energy Star Score Distribution')
# # 在展示的图中，1和100的数量比较高，说明存在异常，由于score是自我报告的，可能存在误差。
# # 有必要进一步分析 能源使用强度(EUI)这个指标。
# # EUI 是 总能源使用量 除以 建筑物的面积（平方英尺）。
# # 这个能源使用量不是自我报告的，因此这可以更客观地衡量建筑物的能源效率。
# # 此外，这不是百分位数，因此绝对值很重要，我们希望它们近似正态分布。

# # 绘制能源使用强度（EUI）直方图
# figsize(8, 8)
# plt.hist(df['Site EUI (kBtu/ft²)'].dropna(), bins=20, edgecolor='black')
# plt.xlabel('Site EUI')
# plt.ylabel('Count')
# plt.title('Site EUI Distribution')
# # 这个结果惊不惊喜，意不意外？这表明我们遇到了另一个问题： 异常值！
# # 由于存在一些分数非常高的建筑物，因此图表非常倾斜。
# # 看起来我们将不得不稍微绕道来处理这些异常值。

# print(df['Site EUI (kBtu/ft²)'].describe())
# print(df['Site EUI (kBtu/ft²)'].dropna().sort_values().tail(10))
# print(df.loc[df['Site EUI (kBtu/ft²)'] == 869265, :])

# 2.2 去除异常值
# 当我们删除异常值时，我们需要小心，我们不会因为它们看起来很奇怪就丢掉测量值。
# 它们可能是我们应该进一步研究的实际现象的结果。 当删除异常值时，我尝试使用极端异常值的定义尽可能保守
first_quartile = df['Site EUI (kBtu/ft²)'].describe()['25%']
third_quartile = df['Site EUI (kBtu/ft²)'].describe()['75%']

iqr = third_quartile - first_quartile
df = df[(df['Site EUI (kBtu/ft²)'] > (first_quartile - 3 * iqr)) &
        (df['Site EUI (kBtu/ft²)'] < (third_quartile + 3 * iqr))]

# figsize(8, 8)
# plt.hist(df['Site EUI (kBtu/ft²)'].dropna(), bins=20, edgecolor='black')
# plt.xlabel('Site EUI')
# plt.ylabel('Count')
# plt.title('Site EUI Distribution')
# # 这幅图看起来好多了，并且接近正常分布，右侧有一条长尾（它有一个正偏斜）。
# # 虽然`Site EUI`可能是一个更客观的衡量标准，但我们的目标仍然是预测能源之星得分，
# # 因此我们将回过头来研究这个变量。 即使分数不是一个好的衡量标准，我们仍然需要预测它，
# # 这就是我们将要做的事情！ 在回到公司的最终报告中，我将指出这可能不是一个客观的衡量标准，
# # 并且使用不同的指标来确定建筑物的效率是个好主意。 此外，如果我们有更多时间参与这个项目，
# # 那么看看分数为1和100的建筑物可能会很有趣，看看它们是否有任何共同之处。

# 2.3 寻找关系
# 为了查看**分类变量 - categorical variables**对分数的影响，
# 我们可以通过**分类变量**的值来绘制**密度图**。
# 密度图还显示单个变量的分布，可以认为是平滑的直方图。
# 如果我们通过为**分类变量**密度曲线着色，这将向我们展示分布如何基于类别变化的。
#
# 我们将制作的第一个图表显示了**分类变量**的分数分布。
# 为了不使图形混乱，我们将图形限制为在数据集中具有超过100个观测值的建筑类型。
types = df.dropna(subset=['score'])
all_types_num = types['Largest Property Use Type'].value_counts()
types = list(all_types_num[all_types_num.values > 90].index)
# print(types)
#
# # 建筑类别分数分布图
# figsize(18, 15)
#
# # 绘制每个建筑物
# for b_type in types:
#     # 选择建筑类型
#     subset = df[df['Largest Property Use Type'] == b_type]
#
#     # 能源之星得分的密度图
#     sns.kdeplot(subset['score'].dropna(), label=b_type, shade=False, alpha=0.8)
#
# plt.xlabel('Energy Star Score', size=20)
# plt.ylabel('Density', size=20)
# plt.title('Density Plot of Energy Star Scores by Building Type', size=20)
# plt.legend()
# # 我们可以看到 建筑类型 对`Energy Star Score`有重大影响。
# # 办公楼往往有较高的分数，而酒店的分数较低。这告诉我们，我们应该在建模中包含建筑类型，
# # 因为它确实对目标有影响。 作为分类变量，我们将不得不对建筑物类型进行one-hot编码。


# # 我们检查另一个分类变量，**自治市镇 - borough**，我们可以制作相同的图表，但这次是由borough着色。
# boroughs = df.dropna(subset=['score'])
# all_boroughs_num = boroughs['Borough'].value_counts()
# boroughs = list(all_boroughs_num[all_boroughs_num.values > 100].index)
# print(boroughs)
# figsize(18, 15)
# for borough in boroughs:
#     subset = df[df['Borough'] == borough]
#     sns.kdeplot(subset['score'].dropna(), label=borough)
# plt.xlabel('Energy Star Score', size=20)
# plt.ylabel('Density', size=20)
# plt.title('Density Plot of Energy Star Scores by Borough', size=28)
# plt.legend()
# 建筑物所在的自治市镇似乎没有像建筑类型那样在分数分布上产生显着差异。
# 尽管如此，将自治市镇纳入分类变量可能是有意义的。


# 2.4 特征与目标之间的相关性
# 为了量化**特征**（变量）和**目标**之间的相关性，我们可以计算[Pearson相关系数]。
# 这是两个变量之间线性关系的强度和方向的度量：
#
# * **- 1** 表示两个变量完全负线性相关，
# * **+1** 表示两个变量完全正线性相关。

# 尽管 **特征**和**目标**之间可能存在非线性关系，相关系数不考虑**特征之间**的相互作用，
# 但**线性关系**是开始探索数据趋势的好方法。 然后，我们可以使用这些值来选择要在我们的模型中使用的特征。

# # 找到所有相关性并排序
# correlations_data = df.corr()['score'].sort_values()

# # 打印最负相关性
# print(correlations_data.head(15),'\n')
#
# # 打印最正相关性
# print(correlations_data.tail(15))
# 特征和目标之间存在几个强烈的负相关。与得分最负相关的是三个不同类别：
# * 能源使用强度 - Energy Use Intensity（EUI），
# * 场地EUI - Site EUI（kBtu /ft²）
# * 天气归一化场地EUI - Weather Normalized Site EUI （kBtu /ft²）（这些在计算方式上略有不同）。
#
# EUI是建筑物使用的能量除以建筑物的平方英尺，用于衡量建筑物的效率，得分越低越好。直观地说，
# 这些相关性是有意义的：**随着EUI的增加，能源之星得分趋于下降**。

# 为了考虑可能的非线性关系，我们可以采用**特征的平方根和自然对数变换**，然后用得分计算相关系数。
# 为了尝试捕捉自治市镇或建筑类型和得分之间任何可能的关系，我们将对这些列进行one-hot编码。


# 选择数字列
numeric_subset = df.select_dtypes('number')

# 使用数字列的平方根和对数创建列
for col in numeric_subset.columns:
    # 跳过the Energy Star Score 这一列
    if col == 'score':
        continue
    else:
        numeric_subset['sqrt_' + col] = np.sqrt(numeric_subset[col])
        numeric_subset['log_' + col] = np.log(numeric_subset[col])

# 选择分类列
categorical_subset = df[['Borough', 'Largest Property Use Type']]

# One hot encode
categorical_subset = pd.get_dummies(categorical_subset)

# 使用concat对两个数据帧进行拼接，确保使用axis = 1来执行列绑定
features = pd.concat([numeric_subset, categorical_subset], axis=1)

# 放弃没有能源之星评分的建筑物
features = features.dropna(subset=['score'])

# # 找到与得分之间的相关性
# correlations = features.corr()['score'].dropna().sort_values()
# # print('corr head: ', correlations.head(15))
# # print('corr tail: ', correlations.tail(15))
# #
# # 转换特征后：
# # * 最强的关系仍然是与能源使用强度（EUI）相关的关系。
# # * 对数和平方根变换似乎没有导致任何更强的关系。
# # * 虽然我们确实看到建筑类型为办公室（`Largest Property Use Type_Office`）建筑与分数略微正相关，但没有强烈的正线性关系。
# # 我们可以使用这些相关性来执行特征选择。


# 现在，让我们绘制数据集中最重要的相关性（就绝对值而言），即`Site EUI（kBtu/ft^2）`。
# 我们可以按照建筑类型为图表着色，以显示它如何影响关系。

# 2.5 双变量图（Two-Variable Plots）
# 为了可视化两个变量之间的关系，我们使用散点图。
# 我们还可以使用诸如标记的颜色或标记的大小等方面包括其他变量。
# 在这里，我们将相互绘制两个数字变量，并使用颜色表示第三个分类变量。

# figsize(12, 10)
# 提取建筑类型
features['Largest Property Use Type'] = df.dropna(subset=['score'])['Largest Property Use Type']

# 限制超过90个观测值的建筑类型（来自之前的代码）,isin()接受一个列表，判断该列中4个属性是否存在
features = features[features['Largest Property Use Type'].isin(types)]

# # 使用seaborn绘制Score与 Log Source EUI 的散点图
# sns.lmplot(x='Site EUI (kBtu/ft²)', y='score',
#            hue='Largest Property Use Type', data=features,
#            scatter_kws={'alpha': 0.8, 's': 60}, fit_reg=False, height=12, aspect=1.2)
# plt.xlabel('Site EUI', size=28)
# plt.ylabel('Energy Star Score', size=28)
# plt.title('Energy Star Score vs Site EUI', size=36)
# # Site EUI与得分之间存在明显的负相关关系。这种关系不是完全线性的,它的相关系数为 -0.7，
# # 但看起来这个特征对于预测建筑物的得分非常重要。

# 2.5.1 Pairs Plot
# 我们可以在几个不同的变量之间建立 Pairs Plot。
# Pairs Plot是一次检查多个变量的好方法，因为它显示了对角线上的变量对和单个变量直方图之间的散点图。

# # 提取要绘制的列
# plot_data = features[['score',
#                       'Site EUI (kBtu/ft²)',
#                       'Weather Normalized Source EUI (kBtu/ft²)',
#                       'log_Total GHG Emissions (Metric Tons CO2e)']]
#
# # 把 inf 换成 nan
# plot_data = plot_data.replace({np.inf: np.nan, -np.inf: np.nan})
#
# # 重命名
# plot_data = plot_data.rename(
#     columns={'Site EUI (kBtu/ft²)': 'Site EUI',
#              'Weather Normalized Source EUI (kBtu/ft²)': 'Weather Norm EUI',
#              'log_Total GHG Emissions (Metric Tons CO2e)': 'log GHG Emissions'})
#
# # 删除 na 值
# plot_data = plot_data.dropna()
#
#
# # 计算某两列之间的相关系数
# def corr_func(x, y, **kwargs):
#     r = np.corrcoef(x, y)[0][1]
#     ax = plt.gca()
#     ax.annotate("r = {:.2f}".format(r),
#                 xy=(.2, .8),
#                 xycoords=ax.transAxes,
#                 size=20)
#
#
# # 创建 pairgrid 对象
# grid = sns.PairGrid(data=plot_data, height=7)
#
# # 上三角是散点图
# grid.map_upper(plt.scatter, color='red', alpha=0.6)
#
# # 对角线是直方图
# grid.map_diag(plt.hist, color='red', edgecolor='black')
#
# # 下三角是相关系数和二维核密度图
# grid.map_lower(corr_func)
# grid.map_lower(sns.kdeplot, cmap=plt.cm.Reds)
#
# plt.suptitle('Pairs Plot of Energy Data', size=40, y=1.0)
# plt.show()
# # 为了解释图中的关系，我们可以查找一行中的变量与一列中的变量相交的位置。例如，
# # 要查找score与log of GHG Emissions之间的关系，我们会查看score列和 log of GHG Emissions行。
# # 在交叉点（左下图），我们看到得分与该变量的相关系数为-0.35。如果我们查看右上图，我们可以看到这种关系的散点图。
# # 要查看Weather EUorm EUI与score的相关性，我们查看Weather EUorm EUI行和score列，可以看到相关系数为-0.67。


# 3. 特征工程和特征选择
# 现在我们已经探索了数据中的趋势和关系，我们可以为我们的模型设计一组函数。
# 我们可以使用EDA的结果来构建特征工程。 特别是，我们从EDA学到了以下知识，
# 可以帮助我们进行特征工程/选择：
#
# * 分数分布因建筑类型而异，并且在较小程度上因行政区而异。 虽然我们将关注数字特征，
# 但我们还应该在模型中包含这两个分类特征。
# * 对特征进行对数变换不会导致特征与分数之间的线性相关性显着增加
#
#
# 在我们进一步讨论之前，我们应该定义什么是特征工程和特征选择！ 这些定义是非正式的，并且有很多重叠，
# 但我喜欢将它们视为两个独立的过程：
# * Feature Engineering:  获取原始数据并提取或创建新特征的过程。
# 这可能意味着需要对变量进行变换，例如自然对数和平方根，或者对分类变量进行one-hot编码，
# 以便它们可以在模型中使用。 一般来说，我认为特征工程是从原始数据创建附加特征。
#
# * Feature Selection:  选择数据中最相关的特征的过程。
# 在特征选择中，我们删除特征以帮助模型更好地总结新数据并创建更具可解释性的模型。
# 一般来说，特征选择是减去特征，所以我们只留下那些最重要的特征。

# 在特征选择中，我们删除了无助于我们的模型学习特征与目标之间关系的特征。
# 这可以帮助模型更好地概括新数据并产生更可解释的模型。
# 一般来说，我认为特征选择为**__subtracting__特征**，因此我们只留下最重要的特征。
#
# 特征工程和选择是迭代过程，通常需要多次尝试才能正确。
# 通常我们会使用建模结果（例如来自随机森林的特征重要性）返回并重做特征选择，
# 或者我们稍后可能会发现需要创建新变量的关系。
# 此外，这些过程通常包含领域知识和数据统计质量的混合。

# Feature engineering and selection 通常具有在机器学习问题上投入的最高回报。
# 要做到正确可能需要一段时间，但通常比用于模型的精确算法和超参数更重要。
# 如果我们不为模型提供正确的数据，那么我们将其设置为失败，我们不应期望它能够学习！


# 3.1 特征工程

# 复制原始数据
features = df.copy()

# 选择数字列
numeric_subset = df.select_dtypes('number')

# 使用数字列的对数创建新列
for col in numeric_subset.columns:
    if col == 'score':
        continue
    else:
        numeric_subset['log_' + col] = np.log(numeric_subset[col])

# 选择分类列
categorical_subset = df[['Borough', 'Largest Property Use Type']]

# One hot encode
categorical_subset = pd.get_dummies(categorical_subset)

# 使用concat对两个数据帧进行拼接，确保使用axis = 1来执行列绑定
features = pd.concat([numeric_subset, categorical_subset], axis=1)
# 也就是说，我们有11319个建筑物，具有109个不同的特征（一列是得分）。
# 并非所有这些特征对于预测得分都很重要，其中一些特征也是多余的，因为它们具有高度相关性。


# 3.2 特征选择（去除共线特征）
# 高共线特征 - [collinear features]在它们之间具有显着的相关系数。
# 例如，在我们的数据集中，Site EUI 和 Weather Norm EUI高度相关，
# 因为它们只是略微不同的计算能源使用强度的方法。
# figsize(12, 10)
# plot_data = df[['Weather Normalized Site EUI (kBtu/ft²)', 'Site EUI (kBtu/ft²)']].dropna()
# plt.plot(plot_data['Site EUI (kBtu/ft²)'], plot_data['Weather Normalized Site EUI (kBtu/ft²)'], 'bo')
# plt.xlabel('Site EUI', size=20)
# plt.ylabel('Weather Norm EUI', size=20)
# plt.title('Weather Norm EUI vs Site EUI, R = %0.4f' %
#           (np.corrcoef(df[['Weather Normalized Site EUI (kBtu/ft²)',
#             'Site EUI (kBtu/ft²)']].dropna(),
#             rowvar=False)[0][1]), size=28)
# 虽然数据集中的变量通常与较小程度相关，但高度共线变量可能是多余的，
# 因为我们只需保留其中一个特征即可为模型提供必要的信息。

# * 删除共线特征是一种通过减少特征数量来降低模型复杂性的方法，可以帮助增加模型泛化。
# * 它还可以帮助我们解释模型，因为我们只需要担心单个变量，
# 例如EUI，而不是 EUI 和 weather normalized EUI如何影响分数。

# 有许多方法可以消除共线特征，例如使用**方差膨胀因子** - [Variance Inflation Factor]
#
# * 我们将使用更简单的度量标准，
# * 并删除相关系数高于某个阈值的特征（不是得分，因为我们想要与得分高度相关的变量！）

# 删除大于指定相关系数的共线特征
features = remove_collinear_features(features, 0.6)

# 删除所有 na 值的列
features = features.dropna(axis=1, how='all')
# 我们的最终数据集现在有64个特征（其中一列是目标）。 这仍然是相当多的，但主要是因为我们有一个one-hot编码的分类变量。 此外，
# * 虽然诸如线性回归之类的模型可能存在大量特征，
# * 但诸如随机森林之类的模型执行隐式特征选择并自动确定在训练期间哪些特征是重要的。

# 有更多的 [特征选择] 方法。一些流行的方法包括主成分分析[PCA]，它将特征转换为保持最大方差的减少数量的维度，
# 或独立成分分析[ICA]，旨在找到一组特征中的独立源。
# 然而，虽然这些方法在减少特征数量方面是有效的，但它们创造了没有物理意义的新特征，因此几乎不可能解释模型。


# 3.3 划分训练集和测试集
# 提取没有得分的建筑物和带有得分的建筑物
# pandas: isna(): 如果参数的结果为NaN，则结果为TRUE，否则是FALSE
no_score = features[features['score'].isna()]

# pandas: notnull() 判断是否NaN
score = features[features['score'].notnull()]

# 将特征和目标分离开
features = score.drop(columns='score')
targets = pd.DataFrame(score['score'])

# 用 nan 替换 inf and -inf （required for later imputation）
# np.inf: 最大值，-np.inf: 最小值
features = features.replace({np.inf: np.nan, -np.inf: np.nan})

# 按照 7：3 的比例划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.3, random_state=22)

# 3.4 建立Baseline
# 在我们开始制作机器学习模型之前建立一个基线是很重要的。
# * 如果我们构建的模型不能胜过基线，那么我们可能不得不承认机器学习不适合这个问题。 这可能是
#    * 因为我们没有使用正确的模型，
#    * 因为我们需要更多的数据，
#    * 或者因为有一个更简单的解决方案，不需要机器学习。

# 建立基线至关重要，因此我们最终可能不会构建机器学习模型，只是意识到我们无法真正解决问题。
#
# 对于回归任务，一个好的基线是为测试集上的所有实例预测目标在训练集上的中值。
# 这很容易实现，并为我们的模型设置了相对较低的标准：如果它们不能比猜测中值更好，
# 那么我们需要重新考虑我们的方法。
baseline_guess = np.median(y_train)  # 选取排序后，中间序号的值
print('The baseline guess is a score of %0.2f' % baseline_guess)
print('Baseline Performance on the test set: MAE = %0.4f' % mae(y_test, baseline_guess))
# 这表明我们对测试集的平均估计偏差约25个百分点。
# 因为得分在1到100之间，这意味着来自基线方法的平均误差约为25％。
# 猜测训练中值的 naive 方法为我们的模型提供了一个低基线！

##### 小结 #####
# 到目前为止，我们完成了
# 1. 清理并格式化原始数据
# 2. 进行探索性数据分析以了解数据集
# 3. 开发了一系列我们将用于模型的特征
# 最后，我们还完成了建立我们可以判断我们的机器学习算法的Baseline的关键步骤。希望你开始了解机器学习管道的每个部分是如何流入下一个管道的：
# 1. 清理数据并将其转换为正确的格式允许我们执行探索性数据分析。
# 2. 然后，EDA在特征工程和选择阶段影响我们的决策。
# 这三个步骤通常按此顺序执行，但我们可能会稍后再回来，根据我们的建模结果进行更多的EDA或特征工程。
# 数据科学是一个迭代过程，我们一直在寻找改进以前工作的方法。

# Save the no scores, training, and testing data
no_score.to_csv('../data/no_score.csv', index=False)
X_train.to_csv('../data/training_features.csv', index=False)
X_test.to_csv('../data/testing_features.csv', index=False)
y_train.to_csv('../data/training_labels.csv', index=False)
y_test.to_csv('../data/testing_labels.csv', index=False)

# 读取格式化后的数据。
train_features = pd.read_csv('../data/training_features.csv')
test_features = pd.read_csv('../data/testing_features.csv')
train_labels = pd.read_csv('../data/training_labels.csv')
test_labels = pd.read_csv('../data/testing_labels.csv')

# figsize(8, 8)
# # Histogram of the Energy Star Score
# plt.style.use('fivethirtyeight')
# plt.hist(train_labels['score'].dropna(), bins=100)
# plt.xlabel('Score')
# plt.ylabel('Number of Buildings')
# plt.title('ENERGY Star Score Distribution')


# 4. 基于性能指标比较几种机器学习模型
# 我们将为我们的监督回归任务构建，训练和评估几种机器学习方法。
# 目标是确定哪个模型最有希望进一步开发（例如超参数调整）。
# 我们使用平均绝对误差比较模型。 猜测得分中值的基线模型平均偏离25分。

# 4.1 输入缺失值
# 标准机器学习模型无法处理缺失值，这意味着我们必须找到一种方法来填充这些缺失值或
# 丢弃任何具有缺失值的特征。 由于我们已经删除了第一部分中缺失值超过50％的特征，
# 因此我们将重点关注这些缺失值，即称为插补的过程。
# 有许多插补方法，但在这里我们将使用相对简单的方法用列的**中位数**替换缺失值。

# 使用中位数填充策略创建一个imputer对象，
# 因为数据有离群点，有大有小，用mean不太合适，用中位数比较合适
imputer = SimpleImputer(strategy='median')

# Train on the training features
imputer.fit(train_features)

# print('Missing values in training features: \n', np.sum(np.isnan(train_features)))
# print('Missing values in testing features: \n', np.sum(np.isnan(test_features)))

# 转换训练数据和测试数据
X = imputer.transform(train_features)
X_test = imputer.transform(test_features)

# np.isnan 数值进行空值检测
print('Missing values in training features: \n', np.sum(np.isnan(X)))
print('Missing values in testing features: \n', np.sum(np.isnan(X_test)))

# Make sure all values are finite
# 确保所有值都是有限的
print(np.where(~np.isfinite(X)))
print(np.where(~np.isfinite(X_test)))

# 4.2 特征缩放
# 在我们构建模型之前要采取的最后一步是[特征缩放]。
# 这是很有必要的，因为特征具有不同的单位，我们希望对特征进行标准化，以使单位不影响算法。
# * [线性回归和随机森林不需要特征缩放]
# * 但其他方法（例如支持向量机和k-最近邻）确实需要它，因为它们考虑了观测之间的欧氏距离。
# 因此，在比较多个算法时，最佳做法是特征缩放。
#
# 有两种[特征缩放]的方法：
# * 对于每个值，减去特征的平均值并除以特征的标准偏差。这称为标准化，并且导致每个特征具有0的均值和1的标准偏差。
# * 对于每个值，减去特征的最小值并除以最大值减去特征的最小值（范围）。这可以确保特征的所有值都在0到1之间，这称为缩放到范围或标准化。

# Create the scaler object with a range of 0-1 - 创建范围为0-1的缩放器对象
scaler = MinMaxScaler(feature_range=(0, 1))

# Fit on the training data
scaler.fit(X)

# 转换训练数据和测试数据
X = scaler.transform(X)
X_test = scaler.transform(X_test)

# Convert y to one-dimensional array (vector)
y = np.array(train_labels).reshape((-1,))
y_test = np.array(test_labels).reshape((-1,))


# 4.3 - 需要评估的模型
# 我们将使用[Scikit-Learn library]比较五种不同的机器学习模型：
# * [线性回归]
# * [支持向量机回归]
# * [随机森林回归]
# * [Gradient Boosting 回归]
# * [K-Nearest Neighbors回归]

# 为了比较模型，我们将主要使用Scikit-Learn默认的模型超参数值。
# 通常这些将表现得很好，但应该在实际使用模型之前进行优化。
# * 首先，我们只想确定每个模型的baseline性能，
# * 然后我们可以选择性能最佳的模型，以便使用超参数调整进行进一步优化。
#   请记住，默认的超参数将启动并运行模型，但几乎总是应该使用某种搜索来调整以找到问题的最佳设置！

# # 线性回归
# lr = LinearRegression()
# lr_mae = fit_and_evaluate(lr, X, y, X_test, y_test)
# print('Linear Regression Performance on the test set: MAE = %0.4f' % lr_mae)
#
# # 支持向量机
# svm = SVR(C=1000, gamma=0.1)
# svm_mae = fit_and_evaluate(svm, X, y, X_test, y_test)
# print('Support Vector Machine Regression Performance on the test set: MAE = %0.4f' % svm_mae)
#
# # 随机森林
# random_forest = RandomForestRegressor(random_state=60)
# random_forest_mae = fit_and_evaluate(random_forest, X, y, X_test, y_test)
# print('Random Forest Regression Performance on the test set: MAE = %0.4f' % random_forest_mae)
#
# # Gradient Boosting Regression
# gradient_boosted = GradientBoostingRegressor(random_state=60)
# gradient_boosted_mae = fit_and_evaluate(gradient_boosted, X, y, X_test, y_test)
# print('Gradient Boosted Regression Performance on the test set: MAE = %0.4f' % gradient_boosted_mae)
#
# # K-Nearest Neighbors Regression
# knn = KNeighborsRegressor(n_neighbors=10)
# knn_mae = fit_and_evaluate(knn, X, y, X_test, y_test)
# print('Knn Regression Performance on the test set: MAE = %0.4f' % knn_mae)

# plt.style.use('fivethirtyeight')
# figsize(30, 10)
# model_comparison = pd.DataFrame({'model': ['Linear Regression',
#                                            'Support Vector Machine',
#                                            'Random Forest',
#                                            'Gradient Boosted',
#                                            'K-Nearest Neighbors'],
#                                  'mae': [lr_mae,
#                                          svm_mae,
#                                          random_forest_mae,
#                                          gradient_boosted_mae,
#                                          knn_mae]})
#
# # 测试集上 mae 的水平条形图
# model_comparison.sort_values('mae', ascending=False).plot(
#                                                     x='model',
#                                                     y='mae',
#                                                     kind='barh',
#                                                     color='red',
#                                                     edgecolor='black')
# # 绘图格式
# plt.xlabel('Mean Absolute Error')
# plt.ylabel('')
# plt.xticks(size=14)
# plt.yticks(size=14)
# plt.title('Model Comparison on Test MAE', size=20)
# 1. 根据运行情况（每次精确结果略有变化），梯度增强回归表现最佳，其次是随机森林。
# 2. 我们不得不承认这不是最公平的比较，因为我们主要使用默认的超参数。 特别是对于支持向量回归器，超参数对性能有重要影响。
# （随机森林和梯度增强方法非常适合开始，因为性能较少依赖于模型设置）。
# 3. 尽管如此，从这些结果中，我们可以得出结论:机器学习是适用的，因为所有模型都明显优于基线！

# 从这里开始，我们将专注于使用超参数调优来优化最佳模型。 鉴于此处的结果，我们将专注于使用GradientBoostingRegressor。
# 这是Gradient Boosted Trees的Scikit-Learn实现Gradient Boosted Trees，在过去的几年中赢得了许多Kaggle比赛Kaggle competitions。
# Scikit-Learn版本通常比XGBoost版本慢，但在这里我们将坚持使用Scikit-Learn，因为语法更为熟悉。 这是 在XGBoost包中使用实现的指南。


# 5. 对最佳模型执行超参数调整，即优化模型
# 在机器学习中，优化模型意味着为特定问题找到最佳的超参数集。

# 5.1 超参数
# * **模型超参数**被认为最好通过机器学习算法来进行设置，在训练之前由数据科学家调整。
#     例如，随机森林中的树木数量，或者K-Nearest Neighbors Regression中使用的邻居数量。
#
# * **模型参数**是模型在训练期间学习的内容，例如线性回归中的权重。

# 选择超参数的问题在于，没有放之四海而皆准的超参数。
# 因此，对于每个新数据集，我们必须找到最佳设置。
# 这可能是一个耗时的过程，但幸运的是，在Scikit-Learn中执行此过程有多种选择。

# [常见设置超参数的做法有]：
# 1. **猜测和检查**：根据经验或直觉，选择参数，一直迭代。
# 2. **网格搜索**：让计算机尝试在一定范围内均匀分布的一组值。
# 3. **随机搜索**：让计算机随机挑选一组值。
# 4. **贝叶斯优化**：使用贝叶斯优化超参数，会遇到贝叶斯优化算法本身就需要很多的参数的困难。
# 5. **在良好初始猜测的前提下进行局部优化**：这就是 MITIE 的方法，它使用 BOBYQA 算法，
#   并有一个精心选择的起始点。由于 BOBYQA 只寻找最近的局部最优解，
#   所以这个方法是否成功很大程度上取决于是否有一个好的起点。
#   在 MITIE 的情下,我们知道一个好的起点，但这不是一个普遍的解决方案，
#   因为通常你不会知道好的起点在哪里。从好的方面来说，这种方法非常适合寻找局部最优解。
# 6. 最新提出的 **LIPO 的全局优化方法**。这个方法没有参数，而且经验证比随机搜索方法好。


# 5.2 使用随机搜索和交叉验证进行超参数调整
# **随机搜索**是指我们选择超参数来评估的方法：
#
# * 我们定义一系列选项，然后随机选择要尝试的组合。
# * 这与网格搜索形成对比，网格搜索评估我们指定的每个组合。

# 通常，当我们对最佳模型超参数的知识有限时，随机搜索会更好，我们可以使用随机搜索缩小选项范围，
# 然后使用更有限的选项范围进行网格搜索。
#
# **交叉验证**是用于评估超参数性能的方法：我们使用K-Fold交叉验证，而不是将训练设置拆分为单独的训练和验证集，以减少我们可以使用的训练数据量。
#
# * 这意味着将训练数据划分为K个折叠，然后进行迭代过程，我们首先在K-1个折叠上进行训练，然后评估第K个折叠的性能。
# * 我们重复这个过程K次，所以最终我们将测试训练数据中的每个例子，关键是每次迭代我们都在测试我们之前没有训练过的数据。
# * 在K-Fold交叉验证结束时，我们将每个K次迭代的平均误差作为最终性能度量，然后立即在所有训练数据上训练模型。
# * 我们记录的性能用于比较超参数的不同组合。

# # 要优化的损失函数
# loss = ['ls', 'lad', 'huber']
#
# # 梯度增强过程中使用的树的数量
# n_estimators = [100, 500, 900, 1100, 1500]
#
# # 树的最大深度
# max_depth = [2, 3, 5, 10, 15]
#
# # 每片叶子的最小样本数
# min_samples_leaf = [1, 2, 4, 6, 8]
#
# # 拆分节点的最小样本数
# min_samples_split = [2, 4, 6, 10]
#
# # 进行拆分时要考虑的最大特征数
# max_features = ['auto', 'sqrt', 'log2', None]
#
# # 定义要进行搜索的超参数网格
# hyperparameter_grid = {'loss': loss,
#                        'n_estimators': n_estimators,
#                        'max_depth': max_depth,
#                        'min_samples_leaf': min_samples_leaf,
#                        'min_samples_split': min_samples_split,
#                        'max_features': max_features}
#
# # Create the model to use for hyperparameter tuning
# model = GradientBoostingRegressor(random_state=22)
#
# # Set up the random search with 4-fold cross validation
# random_cv = RandomizedSearchCV(
#                         estimator=model,  # 模型
#                         param_distributions=hyperparameter_grid,  # 我们定义的参数的分布
#                         cv=4,  # 用于k-fold交叉验证的folds 数量
#                         n_iter=25,  # 不同的参数组合的数量
#                         scoring='neg_mean_absolute_error',  # 评估候选参数时使用的指标
#                         n_jobs=-1,  # 核的数量（-1 时全部使用）
#                         verbose=1,  # 显示信息的数量
#                         return_train_score=True,  # 每一个cross-validation fold 返回的分数
#                         random_state=22  # 修复使用的随机数生成器，因此每次运行都会得到相同的结果
#                         )
#
# # Fit on the training data
# random_cv.fit(X, y)
#
# random_results = pd.DataFrame(random_cv.cv_results_).sort_values('mean_test_score', ascending=False)
# print(random_results.head(10))
# print(random_cv.best_estimator_)
# 使用随机搜索是缩小可能的超参数以尝试的好方法。最初，我们不知道哪种组合效果最好，但这至少缩小了选项的范围。我们可以通过使用随机搜索结果来创建具有超参数的网格来进行网格搜索，这些参数接近于在随机搜索期间最佳的参数。
#
# 但是，我们不会再次评估所有这些设置，而是将**重点放在**单个树林中的树的数量（n_estimators）上。通过仅改变一个超参数，我们可以直接观察它如何影响性能。我们预计会看到树木数量对欠拟合和过拟合的影响。在这里，
#
# * 我们将使用**仅具有n_estimators超参数的网格进行网格搜索**。
# * 我们将评估一系列的树，然后绘制训练和测试性能，以了解增加树的数量对模型的影响。
# * 我们将其他超参数固定为从随机搜索返回的最佳值，以隔离树的数量影响。

# 创建一系列要评估的树
trees_grid = {'n_estimators': [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]}

# 建立模型，lad：最小化绝对偏差
model = GradientBoostingRegressor(loss='lad', max_depth=5,
                                  min_samples_leaf=2,
                                  min_samples_split=10,
                                  max_features='auto',
                                  random_state=22)

# 使用树的范围 和 GradientBoosting 搜索对象
grid_search = GridSearchCV(estimator=model,
                           param_grid=trees_grid,
                           cv=4,
                           scoring='neg_mean_absolute_error',
                           verbose=1,
                           n_jobs=-1,
                           return_train_score=True)

# Fit the grid search
grid_search.fit(X, y)
# print(grid_search.best_estimator_)

# 将结果导入数据框
results = pd.DataFrame(grid_search.cv_results_)

# # 绘制训练误差和测试误差与树木数量的关系图
# figsize(8, 8)
# plt.style.use('fivethirtyeight')
# plt.plot(results['param_n_estimators'], -1 * results['mean_test_score'], label='Testing Error')
# plt.plot(results['param_n_estimators'], -1 * results['mean_train_score'], label='Training Error')
#
# # 横轴是树的个数，纵轴是MAE的误差
# plt.xlabel('Number of Trees')
# plt.ylabel('Mean Abosolute Error')
# plt.legend()
# plt.title('Performance vs Number of Trees')
# 很明显我们的模型**过拟合**了！训练误差明显低于测试误差，这表明模型正在很好地学习训练数据，但是无法推广到测试数据。随着树木数量的增加，
# * 过拟合现象会更严重
# * 测试和训练误差都会减少，但训练误差会减少的更多。
#
# 训练误差和测试误差之间始终存在差异（训练误差始终较低）但如果存在显着差异，
# 我们希望通过**获取更多训练数据或通过超参数调整或正则化降低模型的复杂度**来尝试减少过拟合。对于gradient boosting regressor，一些选项包括:
#
# * 减少树的数量
# * 减少每棵树的最大深度
# * 以及增加叶节点中的最小样本数。
#
# 目前，我们将使用具有最佳性能的模型，并接受它可能过拟合到训练集。
# 根据交叉验证结果，使用800树的最佳模型，实现交叉验证误差在9%以下。这表明能源之星得分的平均交叉验证估计与真实值的误差在9%以内！


# 6. 在测试集上评估最佳模型
# 我们将使用超参数调整中的最佳模型来对测试集进行预测。
# 请记住，我们的模型之前从未见过测试集，所以这个性能应该是模型在现实世界中部署时的表现的一个很好的指标。
#
# 为了比较，我们还可以查看默认模型的性能。 下面的代码创建最终模型，训练它（会有计时），并评估测试集。

# 默认模型
default_model = GradientBoostingRegressor(random_state=22)

# 选择最佳模型
final_model = grid_search.best_estimator_

default_start_time = time.time()
default_model.fit(X, y)
default_end_time = time.time()
print('default delta time: %f s' % (default_end_time - default_start_time))

final_start_time = time.time()
final_model.fit(X, y)
final_end_time = time.time()
print('final delta time: %f s' % (final_end_time - final_start_time))

default_pred = default_model.predict(X_test)
final_pred = final_model.predict(X_test)

print('Default model performance on the test set: MAE = %0.4f.' %
      mae(y_test, default_pred))
print('Final model performance on the test set:   MAE = %0.4f.' %
      mae(y_test, final_pred))

# 最终的模型比基线模型的性能提高了大约10％，但代价是显着增加了运行时间。
# 机器学习通常是一个需要权衡的领域：
#
# * **偏差与方差**
# * **准确性与可解释性**
# * **准确性与运行时间**
# * **以及使用哪种模型**
#
# 最终决定取决于具体情况。 这里，运行时间的增加不是障碍，因为虽然相对差异很大，
# 但训练时间的绝对量值并不显着。 在不同的情况下，权衡可能不一样，
# 因此我们需要考虑我们正在优化的内容以及我们必须使用的限制。


# figsize(8, 8)
# sns.kdeplot(final_pred, label='Predictions')
# sns.kdeplot(y_test, label='Values')
# plt.xlabel('Energy Star Score')
# plt.ylabel('Density')
# plt.title('Test Values and Predictions')
# plt.legend()
# 虽然预测值的密度更接近测试值的中值而不是在100分时的实际峰值，但分布看起来几乎相同。
# 看起来模型在预测极值时可能不太准确，同时预测值 更接近中位数。
# 另一个诊断图是残差的直方图。 理想情况下，我们希望残差是正态分布的，
# 这意味着模型在两个方向（高和低）上误差是相同的。

# figsize(6, 6)
#
# # 计算残差
# residuals = final_pred - y_test
#
# # 绘制残差分布直方图
# plt.hist(residuals, color='red', bins=20, edgecolor='black')
# plt.xlabel('Error')
# plt.ylabel('Count')
# plt.title('Distribution of Residuals')
# 残差接近正态分布，低端有一些明显的异常值。 这些表示模型估计远低于真实值的误差。


# 小结
# 在4，5，6 步我们做了一下几件事：
# 输入缺失值
# 评估和比较几种机器学习方法
# 超参数使用随机搜索和交叉验证来调整机器学习模型
# 评估测试集上的最佳模型
# 结果表明:
# 机器学习适用于我们的问题，最终模型能够将建筑物的能源之星得分的预测误差控制在9.1%以内。
# 我们还看到，超参数调整能够改善模型的性能，尽管在投入的时间方面成本相当高。这是一个很好的提示，
# 正确的特征工程和收集更多数据（如果可能！）比微调模型有更大的回报。
# 我们还观察了运行时间与精度之间的权衡，这是我们在设计机器学习模型时必须考虑的众多因素之一。


# 7. 解释模型结果
# 我们知道我们的模型是准确的，但我们知道为什么它能做出预测？机器学习过程的下一步至关重要：尝试理解模型如何进行预测。
# 实现高精度是很好的，但如果我们能够找出模型能够准确预测的原因，
# 那么我们也可以使用这些信息来更好地理解问题。例如，
#
# * 模型依靠哪些特征来推断能源之星得分？
# * 可以使用此模型进行特征选择，并实现更易于解释的更简单模型吗？
#
# 下面，我们将尝试回答这些问题并从项目中得出最终结论！

# 机器学习经常被批评为一个黑盒子 [criticized as being a black-box]:
# 我们把数据在这边放进去，它在另一边给了我们答案。
# 虽然这些答案通常都非常准确，但该模型并未告诉我们它实际上如何做出预测。
# 这在某种程度上是正确的，但我们可以通过多种方式尝试并发现模型如何“思考”，例如 (LIME)，
# 这种方法试图通过学习围绕预测的线性回归来解释模型预测，这是一个易于解释的模型！

# 我们将探索几种解释模型的方法：
#
# * 特征重要性
# * 本地可解释的模型不可知解释器 (LIME)
# * 检查整体中的单个决策树

# 7.1 特征重要性 - Feature Importances
# 我们可以解释决策树集合的基本方法之一是通过所谓的特征重要性。
# 这些可以解释为最能预测目标的变量。 虽然特征重要性的实际细节非常复杂，
# 但是我们可以使用相对值来比较特征并确定哪些与我们的问题最相关。
#
# 在scikit-learn中，从训练好的树中提取特征重要性非常容易。我们将特征重要性存储在数据框中以分析和可视化它们。

# 将特征重要性提取到数据结构中
feature_results = pd.DataFrame({'feature': list(train_features.columns),
                                'importance': final_model.feature_importances_})
feature_results = feature_results.sort_values('importance', ascending=False).reset_index(drop=True)
# print(feature_results.head(10))

# figsize(25, 20)
# plt.style.use('fivethirtyeight')
#
# # Plot the 10 most important features in a horizontal bar chart
# feature_results.loc[:9, :].plot(x='feature', y='importance',
#                                  edgecolor='k',
#                                  kind='barh', color='blue')
# plt.xlabel('Relative Importance', size=20)
# plt.ylabel('')
# plt.title('Feature Importances from GradientBoosting', size=30)
# plt.show()

# 7.2 使用特征重要性进行特征选择
# 鉴于并非每个特征对于找到分数都很重要，如果我们使用更简单的模型（如线性回归）和
# GradientBoosting 中最重要特征的子集，会发生什么？ 线性回归确实优于基线，但与复杂模型相比表现不佳。
#
# 让我们尝试在**线性回归**中仅使用10个最重要的特征来查看性能是否得到改善。
# 我们还可以限制这些特征并重新评估 GradientBoosting。

# 提取最重要特征的名称
most_important_features = feature_results['feature'][:10]

# 找到与每个特征名称对应的索引
indices = [list(train_features.columns).index(x) for x in most_important_features]

# 数据集中只保留最重要的特征
X_reduced = X[:,indices]
X_test_reduced = X_test[:, indices]

lr = LinearRegression()

# Fit on full set of features - 在全部特征上拟合并测试
lr.fit(X, y)
lr_full_pred = lr.predict(X_test)

# Fit on reduced set of features - 在10个最重要的特征上拟合并测试（即减少后的特征上）
lr.fit(X_reduced, y)
lr_reduced_pred = lr.predict(X_test_reduced)

# Display results
print('Linear Regression Full Results: MAE =    %0.4f.' % mae(y_test, lr_full_pred))
print('Linear Regression Reduced Results: MAE = %0.4f.' % mae(y_test, lr_reduced_pred))
# 可以看出，减少特征并没有改善线性回归的结果！ 事实证明，低重要性特征中的额外信息确实可以提高性能。

#  在10个最重要的特征上拟合并测试（即减少后的特征上） - Fit and test on the reduced set of features
final_model.fit(X_reduced, y)
model_reduced_pred = final_model.predict(X_test_reduced)
print('Gradient Boosted Reduced Results: MAE = %0.4f' % mae(y_test, model_reduced_pred))
# 随着特征数量的减少，模型结果稍差，我们将保留最终模型的所有特征。
# 减少特征数量的初衷是因为我们总是希望构建最简约的模型：
#
# * 即具有足够特征的最简单模型
# * 使用较少特征的模型将更快地训练并且通常更容易解释。
#
# 在现在这种情况下，保留所有特征并不是主要问题，因为训练时间并不重要，我们仍然可以使用许多特征进行解释。


# 7.3 Locally Interpretable Model-agnostic Explanations - 本地可解释的与模型无关的解释
# 我们将使用LIME9 来解释模型所做的个别预测。 LIME是一项相对较新的工作，
# 旨在通过用线性模型近似预测周围的区域来展示机器学习模型的思考方式。

# 我们将试图解释模型在两个例子上得到的预测结果：
#
# * 其中一个例子得到的预测结果非常差
# * 另一个例子得到的预测结果非常好。
#
# 我们将仅仅使用减少后的10个特征来帮助解释。 虽然在10个最重要的特征上训练的模型稍微不准确，
# 但我们通常必须为了可解释性的准确性进行权衡！

# 找到残差
residuals = abs(model_reduced_pred - y_test)

# 提取最差和最好的预测
wrong = X_test_reduced[np.argmax(residuals), :]
right = X_test_reduced[np.argmin(residuals), :]

# 创造一个解释器对象
explainer = lime.lime_tabular.LimeTabularExplainer(
                        training_data=X_reduced,
                        mode='regression',
                        training_labels=y,
                        feature_names=list(most_important_features))

# 显示最差实例的预测值和真实值
print('Wrong Prediction: %0.4f' % final_model.predict(wrong.reshape(1, -1)))
print('Wrong Actual Value: %0.4f' % y_test[np.argmax(residuals)])

print('Right Prediction: %0.4f' % final_model.predict(right.reshape(1, -1)))
print('Right Actual Value: %0.4f' % y_test[np.argmin(residuals)])

# # 最差预测的解释
# wrong_exp = explainer.explain_instance(data_row=wrong,
#                                        predict_fn=final_model.predict)
#
# # 画出预测解释
# figsize(100, 12)
# wrong_exp.as_pyplot_figure()
# plt.title('Explanation of Prediction', size=28)
# plt.xlabel('Effect on Prediction', size=22)

# # Explanation for  correct prediction
# right_exp = explainer.explain_instance(right, final_model.predict, num_features=10)
# right_exp.as_pyplot_figure()
# plt.title('Explanation of Prediction', size=28)
# plt.xlabel('Effect on Prediction', size=22)
# plt.show()


# 7.4 检查单个决策树
# 基于树的集合最酷的部分之一是我们可以查看任何单个估计器（estimator）。
# 虽然我们的最终模型由800个决策树组成，并且查看单个决策树并不表示整个模型，
# 但它仍然允许我们看到决策树如何工作的一般概念。
# 从那里开始，想象出数百棵这些树木可以根据以前树木的错误进行最终预测（这是对梯度增强回归如何工作的显著简化）。

# 提取单个树
single_tree = final_model.estimators_[1][0]
tree.export_graphviz(single_tree,
                     out_file='../images/tree1.dot',
                     rounded=True,
                     feature_names=most_important_features,
                     filled=True)
# 可以使用graphviz包中的dot命令行工具将此.dot文件转换为各种格式，如PDF或PNG。
# 下面这条命令行指令将.dot文件转换为.png图像文件：
# dot -Tpng images/tree1.dot -o images/tree1.png

# 我们可以看到，随着我们增加树的深度，我们将能够更好地拟合数据。
# * 对于小树，每个叶节点中将有许多示例，并且因为模型为节点中的每个示例估计相同的值，所以可能存在更大的错误（除非所有示例具有相同的目标值）。
# * 构造太大的树虽然可能导致过度拟合。
# * 我们可以控制许多超参数，这些参数决定了树的深度和每个叶子中的例子数量。当我们使用交叉验证执行优化时，我们看到了如何选择其中一些超参数。
#
# 虽然我们无法检查模型中的每一棵树，但查看谋个树确实可以让我们了解我们的模型如何进行预测。
# 实际上，这种基于流程图的方法看起来很像人类做出决策，一次回答一个关于单个值的问题。
# 基于决策树的集合简单地采用单个决策树的概念并组合许多个体的预测，以便创建具有比单个估计器更小的方差的模型。
# 树木的集合往往非常准确，也很直观！


# 8.  得出结论 && 记录发现
# 8.1 得出结论
# 机器学习管道的最后部分可能是最重要的：我们需要将我们学到的所有内容压缩成一个简短的摘要，仅突出最重要的发现。
#
# 1. 使用纽约市的能源数据，可以建立一个模型，可以预测建筑物的能源之星得分，误差在10分以内。
# 2. `The Site EUI` and `Weather Normalized Electricity Intensity` 是预测能源之星得分的最相关特征。

# 8.2 记录发现
# 技术项目经常被忽视的部分是文档和报告。 我们可以在世界上做最好的分析，但如果我们没有清楚地传达我们发现的结果，
# 那么它们就不会产生任何影响！当我们记录数据科学项目时，我们会采用所有版本的数据和代码并对其进行打包，以便我们的项目可以被其他数据科学家复制或构建。

# 重要的是要记住：
# 阅读代码的频率高于编写代码
# 如果我们几个月后再回来的话，我们希望确保我们的工作对于其他人和我们自己都是可以理解的,这意味着在代码中添加有用的注释并解释我们的推理。
# 使用笔记本扩展，我们可以隐藏最终报告中的代码，因为虽然很难相信，但并不是每个人都希望在文档中看到一堆Python代码！
# 此外还可以直接下载为pdf或html，然后与他人共享。

# ****** 结语 ******：
#
# 现在是时候结束我们的这个项目了，如果你能从头看到尾，并且在自己的电脑上运行了所有代码，那不管是原作者还是我，都回会感到非常开心。
# 希望现在你对如何处理机器学习项目有了一个宏观的了解。
# 如果你愿意，你可以先修改这个项目并尝试击败现有的模型！
