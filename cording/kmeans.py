import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('C:/Users/kairi/python_practice/convinience_store.csv')

x = df.drop('No', axis=1).values

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(x)

kmeans.cluster_centers_ # それぞれのクラスターの中心座標の確認

cluster = kmeans.predict(x)

df_cluster = df.copy()
df_cluster['cluster'] = cluster
# print(df_cluster.head())

# クラスター結果の考察 -> 教師なし学習は考察する必要がある
# 表を整える　インデックスをクラスター、カラムにラベルにして考察する
df_results = pd.DataFrame()
df_results['cluster 0'] = df_cluster[df_cluster['cluster']==0].mean().tolist()
df_results['cluster 1'] = df_cluster[df_cluster['cluster']==1].mean().tolist()
df_results['cluster 2'] = df_cluster[df_cluster['cluster']==2].mean().tolist()

df_results = df_results.set_index(df_cluster.columns)
df_results = df_results.drop(['No', 'cluster']).T
print(df_results)
