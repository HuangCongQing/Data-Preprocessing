# https://blog.csdn.net/zijinmu69/article/details/82708130


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

ar1 = scio.loadmat("./data/LBPCO_original.mat")
ar1 = ar1['LBPCO_original']
ar2 = scio.loadmat("./data/LBPCO_scaleandstretch.mat")
ar2 = ar2['LBPCO_scaleandstretch']
ar3 = scio.loadmat("./data/LBPCO_scaling.mat")
ar3 = ar3['LBPCO_scaling']
ar4 = scio.loadmat("./data/LBPCO_seamcarving.mat")
ar4 = ar4['LBPCO_seamcarving']
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
plt.scatter(x0[:, 0], x0[:, 1], c = "red", marker='o', label='LBPCO_original')
plt.scatter(x1[:, 0], x1[:, 1], c = "green", marker='*', label='LBPCO_scaleandstretch')
plt.scatter(x2[:, 0], x2[:, 1], c = "blue", marker='+', label='LBPCO_scaling')
plt.scatter(x3[:, 0], x3[:, 1], c = "black", marker='.', label='LBPCO_seamcarving')
plt.xlabel('LBPCO length')
plt.ylabel('LBPCO width')
plt.legend(loc=2)
plt.show()