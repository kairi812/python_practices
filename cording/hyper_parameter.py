# パラメータは学習実行後に獲得できる
# ハイパーパラメータは学習実行前に実行する

# k分割交差検証
# データをk個に分割し、1つをトレーニングデータ、k-1をテストデータとする

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

dataset = load_breast_cancer()

t = dataset.target
x = dataset.data

from sklearn.model_selection import train_test_split

x_train_val, x_test, t_train_val, t_test = train_test_split(x, t, test_size=0.2, random_state=1)
x_train, x_val, t_train, t_val = train_test_split(x_train_val, t_train_val, test_size=0.3, random_state=1)

from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier(max_depth=10, min_samples_split=30, random_state=0)
dtree.fit(x_train, t_train)

print(f'train score: {dtree.score(x_train, t_train)}')
print(f'val score: {dtree.score(x_val, t_val)}')
print(f'test score: {dtree.score(x_test, t_test)}')

# grid-search
# ある程度漏れなくハイパーパラメータの探索を行うことができる
# しかし、場合によって数十～数百パターンの組み合わせを計算する必要があるため時間を要することがある
# 以下、実装
from sklearn.model_selection import GridSearchCV
# estimator, param_grid, cv が必要

estimator = DecisionTreeClassifier(random_state=0)
param_grid = [{
    'max_depth': [5, 10, 15],
    'min_samples_split': [10, 22, 35]
}]
cv = 5

tuned_model = GridSearchCV(estimator=estimator, 
                           param_grid=param_grid,
                           cv=cv,
                           return_train_score=False)
tuned_model.fit(x_train_val, t_train_val)

res = pd.DataFrame(tuned_model.cv_results_).T
# print(res)

tuned_model.best_params_ # 一番いいパラメータ算出
best_model = tuned_model.best_estimator_
print(best_model.score(x_train_val, t_train_val))
print(best_model.score(x_test, t_test))