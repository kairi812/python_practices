from operator import mod
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

dataset = load_boston()
x, t = dataset.data, dataset.target
columns = dataset.feature_names

df = pd.DataFrame(x, columns=columns)
df['Target'] = t

t = df['Target'].values # ndarray
x = df.drop(labels=['Target'], axis=1).values # Targetの縦列を削除したものを格納

x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.3, random_state=0)

from sklearn.linear_model import LinearRegression

model = LinearRegression() # create instance
model.fit(x_train, t_train) # learning

model.coef_ # weight
model.intercept_ # bias

# plt.figure(figsize=(10, 7))
# plt.bar(x=columns, height=model.coef_)
# plt.show()

print(f'train score: {model.score(x_train, t_train)}')
print(f'test score: {model.score(x_test, t_test)}')

# 予測と比較
y = model.predict(x_test)
print(f'予測値: {y[1]}')
print(f'目標値: {t_test[1]}')

# 過学習(over fitting)を抑制する方法
# データのサンプル数を増やす
# ハイパーパラメータチューニング
# ほかのアルゴリズム使用

df2 = pd.read_csv('C:/Users/kairi/python_practice/regression_pls.csv')
t2 = df2['Target'].values
x2 = df2.drop('Target', axis=1).values

x2_train, x2_test, t2_train, t2_test = train_test_split(x2, t2, test_size=0.3, random_state=0)

model2 = LinearRegression()
model2.fit(x2_train, t2_train)

print(f'train score: {model2.score(x2_train, t2_train)}')
print(f'test score: {model2.score(x2_test, t2_test)}')

# print(df2.corr()) # 相関関係を見る
df2_corr = df2.corr()
# plt.figure(figsize=(12, 8))
# sns.heatmap(df2_corr.iloc[:20, :20], annot=True) # heatmap
# plt.show()

# sns.jointplot(x='x1', y='x16', data=df2) # あるラベルの相関を見る
# plt.show()

# PLS
from sklearn.cross_decomposition import PLSRegression

pls = PLSRegression(n_components=11)
pls.fit(x2_train, t2_train)

print(f'train score: {pls.score(x2_train, t2_train)}')
print(f'test score: {pls.score(x2_test, t2_test)}')