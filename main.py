import numpy as np
import pandas as pd
import os
import time
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
# plt.style.use('ggplot')
import lightgbm as lgb
import xgboost as xgb
# import plotly.offline as py
# py.init_notebook_mode(connected=True)
import warnings
warnings.filterwarnings("ignore")
# import time
# import datetime
# print(plt.style.available)
# plt.style.use('seaborn-v0_8')
plt.rcParams.update({'font.size': 4})
sns.set(font_scale=2)
pd.set_option('display.max_columns', 500)

train=pd.read_csv('./train.csv')
# train = reduce_mem_usage(train)
test=pd.read_csv('./test.csv')
# test = reduce_mem_usage(test)

#合并训练集和测试集，便于进行统一数据预处理
X = pd.concat([train.drop("HasDetections", axis=1),test], axis=0)
y = train[['HasDetections']]

X.info()

#分割数值型数据和分类数据
numeric_ = X.select_dtypes(exclude=['object']).copy()
# numeric_.columns

null_num_var = ['RtpStateBitfield ','IsSxsPassiveMode','AVProductsInstalled','AVProductsEnabled','IeVerIdentifier','Census_OEMNameIdentifier',
                'Census_OEMModelIdentifier', 'Census_ProcessorManufacturerIdentifier', 'Census_ProcessorModelIdentifier', 'Census_InternalBatteryType',
                'Census_InternalBatteryNumberOfCharges', 'Census_OSInstallLanguageIdentifier','Census_OSUILocaleIdentifier','Census_IsFlightingInternal',
               'Census_ThresholdOptIn','Census_FirmwareManufacturerIdentifier','Census_FirmwareVersionIdentifier','Census_IsWIMBootEnabled','Wdft_RegionIdentifier']
cont_num_var = []
for i in numeric_.columns:
    if i not in null_num_var:
        cont_num_var.append(i)

#分割分类数据
cat_train = X.select_dtypes(include=['object']).copy()

#对数值型数据进行统计特征分析
fig = plt.figure(figsize=(18,16))
for index,col in enumerate(cont_num_var):
    plt.subplot(7,5,index+1)
    sns.distplot(numeric_.loc[:,col].dropna(), kde=False)
fig.tight_layout(pad=1.0)

fig = plt.figure(figsize=(14,15))
for index,col in enumerate(cont_num_var):
    plt.subplot(7,5,index+1)
    sns.boxplot(y=col, data=numeric_.dropna())
fig.tight_layout(pad=1.0)

#分析数值型特征关联矩阵
plt.figure(figsize=(14,12))
correlation = numeric_.corr()
sns.heatmap(correlation, mask = correlation <0.8, linewidth=0.9, cmap='Blues')

#和预测变量的关联程度
numeric_train = train.select_dtypes(exclude=['object'])
correlation = numeric_train.corr()
correlation[['HasDetections']].sort_values(['HasDetections'], ascending=False)

# 特征与预测变量关联散点图
# fig = plt.figure(figsize=(20,20))
# for index in range(len(numeric_train.columns)):
#     plt.subplot(10,4,index+1)
#     sns.scatterplot(x=numeric_train.iloc[:,index], y='HasDetections', data=numeric_train.dropna())
# fig.tight_layout(pad=1.0)

#删除高度相关的特征

#有太多缺失值的特征
plt.figure(figsize=(25,8))
plt.title('Number of missing rows')
missing_count = pd.DataFrame(X.isnull().sum(), columns=['sum']).sort_values(by=['sum'],ascending=False).head(20).reset_index()
missing_count.columns = ['features','sum']
plt.xticks(rotation=90)
sns.barplot(x='features',y='sum', data = missing_count)
X.drop(['PuaMode','Census_ProcessorClass'], axis=1, inplace=True)

#去掉有过多单一值的特征
cat_col = X.select_dtypes(include=['object']).columns
overfit_cat = []
for i in cat_col:
    counts = X[i].value_counts()
    zeros = counts.iloc[0]
    if zeros / len(X) * 100 > 96:
        overfit_cat.append(i)

overfit_cat = list(overfit_cat)
X = X.drop(overfit_cat, axis=1)

num_col = X.select_dtypes(exclude=['object']).drop(['MSSubClass'], axis=1).columns
overfit_num = []
for i in num_col:
    counts = X[i].value_counts()
    zeros = counts.iloc[0]
    if zeros / len(X) * 100 > 96:
        overfit_num.append(i)

overfit_num = list(overfit_num)
X = X.drop(overfit_num, axis=1)

print("Categorical Features with >96% of the same value: ",overfit_cat)
print("Numerical Features with >96% of the same value: ",overfit_num)


