# https://blog.csdn.net/zijinmu69/article/details/82708130
# https://blog.csdn.net/guofei_fly/article/details/85485173

# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.cluster import KMeans
# #from sklearn import datasets
# from sklearn.datasets import load_iris


import scipy.io as scio

import pandas as pd


#Method 1
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
#from sklearn import datasets
from sklearn.datasets import load_iris

ar1 = scio.loadmat("./data/BSIF_original.mat")
ar1 = ar1['BSIF_original']
ar2 = scio.loadmat("./data/BSIF_scaleandstretch.mat")
ar2 = ar2['BSIF_scaleandstretch']
ar3 = scio.loadmat("./data/BSIF_scaling.mat")
ar3 = ar3['BSIF_scaling']
ar4 = scio.loadmat("./data/BSIF_seamcarving.mat")
ar4 = ar4['BSIF_seamcarving']
# print(data1)
# print (type(data1))


np1 = np.hstack((ar1,ar2,ar3,ar4)) 


estimator = KMeans(n_clusters=4)#构造聚类器
estimator.fit(np1)#聚类
label_pred = estimator.labels_  #获取聚类标签

#绘制k-means结果
x0 = np1[label_pred == 0]
x1 = np1[label_pred == 1]
x2 = np1[label_pred == 2]
x3 = np1[label_pred == 3]
# plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0) # 图片如何充满整个画布???
plt.scatter(x0[:, 0], x0[:, 1], c = "red", marker='o', label='BSIF_original')
plt.scatter(x1[:, 0], x1[:, 1], c = "green", marker='*', label='BSIF_scaleandstretch')
plt.scatter(x2[:, 0], x2[:, 1], c = "blue", marker='+', label='BSIF_scaling')
plt.scatter(x3[:, 0], x3[:, 1], c = "black", marker='.', label='BSIF_seamcarving')
plt.xlabel('BSIF length')
plt.ylabel('BSIF width')
plt.legend(loc=2)
plt.show()