import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.datasets import load_iris

dataset = load_iris()
columns_name = dataset.feature_names

x = dataset.data
t = dataset.target

from sklearn.model_selection import train_test_split

x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.3, random_state=0)

from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(random_state=0)

tree.fit(x_train, t_train)

print(f'train score: {tree.score(x_train, t_train)}')
print(f'test score: {tree.score(x_test, t_test)}')

tree.predict(x_test)

import graphviz
from sklearn.tree import export_graphviz

# 現時点では表示できないので jupyter などで書くとわかりやすい
dot_data = export_graphviz(tree)
graph_tree = graphviz.Source(dot_data)

feature_importance = tree.feature_importances_

y = columns_name
width = feature_importance

# plt.barh(y=y, width=width)
# plt.show()

from sklearn.svm import SVC

model = SVC()
model.fit(x_train, t_train)
print(f'train score: {model.score(x_train, t_train)}')
print(f'test score: {model.score(x_test, t_test)}')

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x_train)
x_train_std = scaler.transform(x_train)
x_test_std = scaler.transform(x_test)

model_std = SVC(C=1, gamma=0.01)
model_std.fit(x_train_std, t_train)
print(f'train score: {model.score(x_train, t_train)}')
print(f'test score: {model.score(x_test, t_test)}')
print('================================')
print(f'train score: {model_std.score(x_train_std, t_train)}')
print(f'test score: {model_std.score(x_test_std, t_test)}')

