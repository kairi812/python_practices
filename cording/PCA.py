import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

dataset = load_iris()

x = dataset.data
t = dataset.target # 教師なし学習なので目標値である t は使用しない
feature_names = dataset.feature_names

# 主成分分析を使って高次のものを2次元に落とし込む
from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=0) # n_components:次元
pca.fit(x)
pca.get_covariance()

# データ変換されていないので主成分へ写像を行なう -> transform メソッド
x_transformed = pca.transform(x)
res = pd.DataFrame(x_transformed, columns=['第一主成分', '第二主成分'])
print(res)
# それぞれの列が保持する元のデータの情報の割合を寄与率 (Proportion of the variance) と呼ぶ
print(pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1])
# この例なら 97% 程度の割合で元のデータ情報を保持したまま次元削減できている
# 3 % は損失した
# 元のデータをどの程度再現できているかを確認すること（つまり寄与率を確認すること）は大事

sns.scatterplot(x_transformed[:,0], x_transformed[:,1], hue=t, palette=sns.color_palette(n_colors=3))

# 主成分分析では必ず標準化を行なう
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

x_scaled_transformed = pca.fit_transform(x_scaled)
res = pd.DataFrame(x_scaled_transformed, columns=['第一主成分', '第二主成分'])
print(pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1])

sns.scatterplot(x_scaled_transformed[:,0], x_scaled_transformed[:,1], hue=t, palette=sns.color_palette(n_colors=3))
